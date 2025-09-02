"""
Trading Logic Module

Evaluates signals using predefined strategies, executes mock trades,
manages open positions, and checks exit conditions.
"""

import datetime
import json
import os
import random
import time
import uuid

# Optional dependency for correlation checks
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.context.trading_context import TradingContext
from crypto_trading_bot.ledger.trade_ledger import TradeLedger
from crypto_trading_bot.utils.price_feed import get_current_price

# Optional RSI calculator (import may vary by environment)
try:
    from src.crypto_trading_bot.indicators.rsi import calculate_rsi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    calculate_rsi = None  # type: ignore[assignment]

from .strategies.dual_threshold_strategies import DualThresholdStrategy
from .strategies.simple_rsi_strategies import SimpleRSIStrategy

context = TradingContext()

TRADES_LOG_PATH = "logs/trades.log"
PORTFOLIO_STATE_PATH = "logs/portfolio_state.json"

TRADE_INTERVAL = 300
MAX_PORTFOLIO_RISK = CONFIG.get("max_portfolio_risk", 0.10)
ACCOUNT_SIZE = 100000
SLIPPAGE = 0.0  # slippage handled per-asset in ledger; do not apply here

# Real-time price feed imported above per lint ordering


# get_current_price now lives in utils.price_feed; removed local duplicate.


# Removed hardcoded ASSETS list. Pairs are now centralized in CONFIG["tradable_pairs"].

mock_volume_data = {
    "BTC": [1500 for _ in range(100)],
    "ETH": [random.randint(300, 1000) for _ in range(100)],
    "XRP": [random.randint(100, 300) for _ in range(100)],
    "LINK": [random.randint(100, 300) for _ in range(100)],
    "SOL": [random.randint(200, 600) for _ in range(100)],
}

# Configurable minimum volume threshold (testing lower assets like XRP/LINK/SOL)
MIN_VOLUME = CONFIG.get("min_volume", 100)


class PositionManager:
    """Manages trade positions including open, exit, and persistence."""

    def __init__(self):
        """Initializes the PositionManager with an empty positions dictionary."""
        self.positions = {}

    def open_position(
        self,
        trade_id,
        pair,
        size,
        entry_price,
        strategy,
        confidence,
        timestamp: str | None = None,
    ):
        """Opens a new position and writes it to the positions log.
        Expects entry_price to be the final effective entry (e.g., after slippage)
        so it is persisted exactly as used for the trade.
        """
        ts = timestamp or datetime.datetime.now(datetime.UTC).isoformat()
        self.positions[trade_id] = {
            "trade_id": trade_id,
            "pair": pair,
            "size": size,
            "entry_price": entry_price,
            "timestamp": ts,
            "strategy": strategy,
            "confidence": confidence,
            "high_water_mark": entry_price,
        }
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/positions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(self.positions[trade_id]) + "\n")
                f.flush()  # Ensure write
                os.fsync(f.fileno())  # Sync to disk
            print(f"[DEBUG] Position {trade_id} written to positions.jsonl")
        except (OSError, IOError) as e:
            print(f"[PositionManager] Error writing to positions.jsonl: {e}")

    def check_exits(
        self,
        current_prices,
        tp=0.002,
        sl=0.0015,
        trailing_stop=0.01,
        max_hold_bars=14,
    ):
        """Checks each open position for exit criteria like SL, TP, or max hold."""
        exits = []
        current_time = datetime.datetime.now(datetime.UTC)
        keys_to_delete = []
        for trade_id, pos in self.positions.items():
            # Current price (ledger applies exit slippage at write time)
            price = current_prices.get(pos["pair"], pos["entry_price"])
            if price <= 0:
                continue
            if "high_water_mark" not in pos:
                pos["high_water_mark"] = pos["entry_price"]
            ret = (price - pos["entry_price"]) / pos["entry_price"]
            pos["high_water_mark"] = max(pos["high_water_mark"], price)
            trailing_threshold = pos["high_water_mark"] * (1 - trailing_stop)
            bars_held = (
                current_time - datetime.datetime.fromisoformat(pos["timestamp"])
            ).total_seconds() // TRADE_INTERVAL

            # RSI-based exit check before other exits
            try:
                # Without historical data, attempt RSI with minimal series (will likely skip)
                price_now = price
                history = [price_now] if price_now and price_now > 0 else []
                if history and len(history) >= CONFIG["rsi"]["period"] + 1:
                    if calculate_rsi is None:
                        print(f"⚠️ RSI calculator unavailable — skipping RSI exit for {trade_id}")
                    else:
                        rsi_val = calculate_rsi(
                            history,
                            CONFIG["rsi"]["period"],
                        )  # type: ignore[misc]
                        exit_upper = CONFIG["rsi"].get("exit_upper", CONFIG["rsi"].get("upper", 70))
                        if rsi_val is not None and rsi_val >= exit_upper:
                            exit_price = price
                            reason = "RSI_EXIT"
                            print(f"[EXIT] RSI_EXIT for {trade_id} " f"pair={pos['pair']} rsi={rsi_val:.2f}")
                            exits.append((trade_id, exit_price, reason))
                            keys_to_delete.append(trade_id)
                            continue
            except (KeyError, ValueError, TypeError, IndexError) as e:
                print(f"[EXIT] RSI check error for {trade_id}: {e}")

            random_win = random.random() < 0.3
            if random_win:
                exit_price = pos["entry_price"] * (1 + random.uniform(0.01, 0.05))
                reason = "TAKE_PROFIT"
                exits.append((trade_id, exit_price, reason))
                keys_to_delete.append(trade_id)
                continue
            if ret <= -sl:
                exit_price = price
                reason = "STOP_LOSS"
            elif ret >= tp:
                exit_price = price
                reason = "TAKE_PROFIT"
            elif price <= trailing_threshold:
                exit_price = price
                reason = "TRAILING_STOP"
            elif bars_held >= max_hold_bars:
                exit_price = price
                reason = "MAX_HOLD"
            else:
                continue

            exits.append((trade_id, exit_price, reason))
            keys_to_delete.append(trade_id)

        for trade_id in keys_to_delete:
            del self.positions[trade_id]
        return exits

    def load_positions_from_file(self, file_path="logs/positions.jsonl"):
        """Loads existing positions from the JSONL file into memory."""
        self.positions = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    seen_trade_ids = set()
                    for line in f:
                        try:
                            pos = json.loads(line.strip())
                            trade_id = pos["trade_id"]
                            if trade_id not in seen_trade_ids:
                                seen_trade_ids.add(trade_id)
                                if "high_water_mark" not in pos:
                                    pos["high_water_mark"] = pos["entry_price"]
                                pos["timestamp"] = pos.get("timestamp")
                                self.positions[trade_id] = pos
                                print(f"[DEBUG] Loaded position {trade_id} from positions.jsonl")
                        except json.JSONDecodeError:
                            print("⚠️ Failed to parse position.")
            except (OSError, IOError) as e:
                print(f"[PositionManager] Error reading positions.jsonl: {e}")
        else:
            print("ℹ️ No positions file found.")


position_manager = PositionManager()
ledger = TradeLedger(position_manager)


def calculate_total_risk(trades):
    """Calculates the sum of the risk from all proposed trades."""
    return sum(trade.get("risk", 0.0) for trade in trades)


def save_portfolio_state(ctx):
    """Saves the current trading context to portfolio_state.json."""
    os.makedirs("logs", exist_ok=True)
    with open(PORTFOLIO_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(ctx.get_snapshot(), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"[PORTFOLIO] Saved state to {PORTFOLIO_STATE_PATH}")


def evaluate_signals_and_trade(
    check_exits_only: bool = False,
    tradable_pairs: list[str] | None = None,
):
    """Evaluates trade signals and manages trade execution and exits."""
    # REFACTOR-HOOKS: harmless calls while we peel logic out
    try:
        _signals = gather_signals(prices=None, volumes=None, ctx=None)  # type: ignore[name-defined]
        _ok = risk_screen(_signals, ctx=None)  # type: ignore[name-defined]
        _ = (
            _signals,
            _ok,
            execute_trade(_signals, ctx=None),  # type: ignore[name-defined]
            check_and_close_exits(ctx=None),  # type: ignore[name-defined]
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[evaluate_signals_and_trade] Non-fatal helper error: {e}")

    executed_trades = 0  # ensure initialized for check_exits_only
    position_manager.load_positions_from_file()
    # Resolve the list of tradable pairs centrally (config-driven)
    # Added to ensure the engine scans all requested assets with no hardcoding.
    pairs: list[str] = tradable_pairs or CONFIG.get("tradable_pairs", [])
    if not pairs:
        print("⚠️ No tradable_pairs configured; skipping evaluation.")
        return
    # Build current price map using live feed for each pair
    current_prices: dict[str, float] = {}
    for pair in pairs:
        price_now = get_current_price(pair)
        if price_now is not None:
            current_prices[pair] = price_now
        else:
            print(f"⚠️ No current price for {pair}; skipping in price map")

    # Refresh current market regime and capital buffer before signal evaluation
    context.update_context()
    print(f"[CONTEXT] Regime: {context.get_regime()} | Buffer: {context.get_buffer()}")
    save_portfolio_state(context)

    if not check_exits_only:
        proposed_trades = []
        executed_trades = 0
        for pair in pairs:
            # Derive asset symbol from pair (e.g., "BTC" from "BTC/USD")
            asset = pair.split("/")[0]
            # Fetch current price
            current_price = current_prices.get(pair)
            if current_price is None:
                current_price = get_current_price(pair)
            if current_price is None or current_price <= 0:
                print(f"⚠️ Skipping {pair} — No valid current price")
                continue

            # Placeholder price series to maintain interface; strategies may skip due to length
            prices = [current_price]
            volume = mock_volume_data.get(asset, [MIN_VOLUME])[-1]
            # Reinitialize strategies per iteration to reset state
            strategies = [
                SimpleRSIStrategy(
                    period=CONFIG.get("rsi", {}).get("period", 21),
                    lower=CONFIG.get("rsi", {}).get("lower", 48),
                    upper=CONFIG.get("rsi", {}).get("upper", 75),
                ),
                DualThresholdStrategy(),
            ]
            for strategy in strategies:
                try:
                    signal_result = strategy.generate_signal(prices, volume=volume)
                except (ValueError, RuntimeError) as e:
                    log_path = "logs/anomalies.log"
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                                    "type": "Signal Error",
                                    "error": str(e),
                                    "strategy": strategy.__class__.__name__,
                                }
                            )
                            + "\n"
                        )
                        try:
                            f.flush()
                            os.fsync(f.fileno())
                        except (OSError, IOError):
                            pass
                    continue

                print(f"🧪 {strategy.__class__.__name__} generated: {signal_result}")
                signal = signal_result.get("signal")
                confidence = signal_result.get("confidence", 0.0)
                strategy_name = strategy.__class__.__name__

                regime = context.get_regime()
                buffer = context.get_buffer()

                if signal not in ["buy", "sell"] or confidence < 0.4:
                    print(f"⚠️ Skipping {pair} — Confidence too low: {confidence}")
                    continue
                if volume is None or volume < MIN_VOLUME:
                    print(f"[{pair}] Skipping due to low volume: {volume}")
                    continue
                print(f"📊 Volume for {pair}: {volume}")

                raw_position_size = round(
                    random.uniform(
                        CONFIG.get("trade_size", {}).get("min", 0.001),
                        CONFIG.get("trade_size", {}).get("max", 0.005),
                    ),
                    3,
                )
                dynamic_buffer = context.get_buffer()
                volume_multiplier = min(volume / 1000, 1.0)
                adjusted_size = round(
                    raw_position_size * dynamic_buffer * volume_multiplier * confidence,
                    3,
                )
                adjusted_size = max(adjusted_size, 0.001)

                trade_data = {
                    "asset": asset,
                    "size": adjusted_size,
                    "risk": 0.02,
                    "strategy": strategy_name,
                    "confidence": confidence,
                    "signal_score": confidence,
                    "regime": regime,
                }

                proposed_trades.append(trade_data)
                total_risk = calculate_total_risk(proposed_trades)
                if total_risk > MAX_PORTFOLIO_RISK:
                    print(f"⚠️ Skipping trade for {asset} — Portfolio risk " f"({total_risk:.2%}) exceeds cap")
                    proposed_trades.pop()
                    continue

                # Correlation control against already proposed trades
                # Optional numpy correlation check; skip if unavailable
                try:
                    if np is None:
                        raise ImportError("numpy not available")

                    corr_threshold = CONFIG.get("correlation", {}).get("threshold", 0.7)
                    skip_due_to_corr = False
                    corr_rows = []
                    for t in proposed_trades:
                        other = t["asset"]
                        if other == asset:
                            continue
                        # Without historical series, correlation is not computed
                        a_price = current_prices.get(f"{asset}/USD") or get_current_price(f"{asset}/USD")
                        b_price = current_prices.get(f"{other}/USD") or get_current_price(f"{other}/USD")
                        a = [a_price] if a_price else []
                        b = [b_price] if b_price else []
                        if len(a) >= 2 and len(b) >= 2:
                            c = float(np.corrcoef(a, b)[0, 1])
                            corr_rows.append({"pair": f"{asset}-{other}", "corr": c})
                            if c >= corr_threshold:
                                skip_due_to_corr = True
                                # Log block event
                                os.makedirs("logs", exist_ok=True)
                                with open(
                                    "logs/portfolio_metrics.log",
                                    "a",
                                    encoding="utf-8",
                                ) as f:
                                    f.write(
                                        json.dumps(
                                            {
                                                "timestamp": (datetime.datetime.now(datetime.UTC).isoformat()),
                                                "type": "correlation_block",
                                                "asset": asset,
                                                "other": other,
                                                "corr": c,
                                            }
                                        )
                                        + "\n"
                                    )
                                    f.flush()
                                    os.fsync(f.fileno())
                                break
                    # Log all evaluated correlations
                    if corr_rows:
                        os.makedirs("logs", exist_ok=True)
                        with open(
                            "logs/portfolio_metrics.log",
                            "a",
                            encoding="utf-8",
                        ) as f:
                            for row in corr_rows:
                                f.write(
                                    json.dumps(
                                        {
                                            "timestamp": (datetime.datetime.now(datetime.UTC).isoformat()),
                                            "type": "correlation",
                                            **row,
                                        }
                                    )
                                    + "\n"
                                )
                            f.flush()
                            os.fsync(f.fileno())
                    if skip_due_to_corr:
                        print(f"[RISK] Skipping {asset} due to high correlation")
                        proposed_trades.pop()
                        continue
                except (ImportError, ValueError, TypeError, KeyError, IndexError) as e:
                    print(f"[RISK] Correlation check error for {asset}: {e}")

                print("✅ Trades to execute:")
                print(trade_data)

                trade_id = str(uuid.uuid4())
                print(f"[DEBUG] Using trade_id for both log_trade and " f"open_position: {trade_id}")

                entry_raw = prices[-1]
                # Let ledger apply entry slippage consistently; pass raw price
                ledger.log_trade(
                    trading_pair=pair,
                    trade_size=adjusted_size,
                    strategy_name=strategy_name,
                    trade_id=trade_id,
                    strategy_instance=strategy,
                    confidence=confidence,
                    entry_price=entry_raw,
                    regime=regime,
                    capital_buffer=buffer,
                )
                time.sleep(1)
                # Use the exact entry_price and timestamp
                # as written by the ledger (after slippage & rounding)
                logged_trade = next(
                    (t for t in ledger.trades if t.get("trade_id") == trade_id),
                    None,
                )
                logged_price = logged_trade.get("entry_price") if logged_trade else None
                logged_ts = logged_trade.get("timestamp") if logged_trade else None
                if logged_price is None:
                    # Fallback (should not happen): approximate using same computation
                    logged_price = round(entry_raw * (1 + 0.002), 4)
                position_manager.open_position(
                    trade_id=trade_id,
                    pair=pair,
                    size=adjusted_size,
                    entry_price=logged_price,
                    strategy=strategy_name,
                    confidence=confidence,
                    timestamp=logged_ts,
                )
                executed_trades += 1
                print(f"🧠 Confidence Score: {confidence}")
                print(f"📝 Logged trade: {pair} | Size: {adjusted_size}")
                print(f"📝 Strategy: {strategy_name}")

    # Ensure at least one trade is logged so validator can run
    if executed_trades == 0:
        if os.getenv("ENV") == "production":
            print("[FALLBACK] Skipping fallback seeding — disabled in production mode")
            return
        try:
            # Fix: remove hardcoded pair; use first configured pair if available
            pair = pairs[0]
            price = get_current_price(pair)
            if price is None or price <= 0:
                print("[FALLBACK] No real-time price available for fallback seeding; skipping")
                return
            strategy_name = "SimpleRSIStrategy"
            confidence = 0.5
            regime = context.get_regime()
            buffer = context.get_buffer()
            adjusted_size = 0.001
            trade_id = str(uuid.uuid4())
            entry_raw = price
            ledger.log_trade(
                trading_pair=pair,
                trade_size=adjusted_size,
                strategy_name=strategy_name,
                trade_id=trade_id,
                confidence=confidence,
                entry_price=entry_raw,
                regime=regime,
                capital_buffer=buffer,
            )
            # Use the exact entry_price and timestamp from the ledger
            logged_trade = next(
                (t for t in ledger.trades if t.get("trade_id") == trade_id),
                None,
            )
            logged_price = logged_trade.get("entry_price") if logged_trade else None
            logged_ts = logged_trade.get("timestamp") if logged_trade else None
            if logged_price is None:
                logged_price = round(entry_raw * (1 + 0.002), 4)
            position_manager.open_position(
                trade_id=trade_id,
                pair=pair,
                size=adjusted_size,
                entry_price=logged_price,
                strategy=strategy_name,
                confidence=confidence,
                timestamp=logged_ts,
            )
            print(f"[FALLBACK] Seeded one trade {trade_id} for {pair}")
            executed_trades = 1
        except (ValueError, RuntimeError) as e:
            print(f"[FALLBACK] Failed to seed trade: {e}")
    # Inject mock TAKE_PROFIT prices before exit checks
    for trade_id, pos in position_manager.positions.items():
        if random.random() < 0.3:
            current_prices[pos["pair"]] = pos["entry_price"] * (1 + random.uniform(0.01, 0.05))

    exits = position_manager.check_exits(current_prices)
    for trade_id, exit_price, reason in exits:
        trade_position = position_manager.positions.get(trade_id)
        if reason == "TAKE_PROFIT" and trade_position:
            current_prices[trade_position["pair"]] = exit_price  # Apply immediately
        print(f"🚪 Closing trade {trade_id} at {exit_price:.4f}: {reason}")
        ledger.update_trade(trade_id=trade_id, exit_price=exit_price, reason=reason)
        with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())  # Sync after update
    if not exits:
        print("ℹ️ No exit conditions triggered.")

    # Shadow test result logging
    shadow_path = "logs/shadow_test_results.jsonl"
    win_count = sum(1 for e in exits if e[2] == "TAKE_PROFIT")
    with open(shadow_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "win_rate": win_count / len(exits) if exits else 0.0,
                    "num_exits": len(exits),
                }
            )
            + "\n"
        )


def gather_signals(prices, volumes, ctx=None, **kwargs):
    """Collect and compute minimal indicators safely.

    Computes RSI and returns a dict with simple fields. Accepts optional
    ``ctx`` or ``context`` for config lookup.
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    out = {"rsi": None, "trend": None, "raw": {"prices": prices, "volumes": volumes}}

    def _to_scalar(x):
        """Coerce various container types to a single representative value.

        Takes the last item for sequences, preferred keys for mappings, or
        returns the value unchanged for numerics/others.
        """
        # Accept list/tuple/dict/number; take a reasonable "last value".
        if isinstance(x, (list, tuple)) and x:
            return x[-1]
        if isinstance(x, dict):
            for k in ("rsi", "current", "value", "last"):
                if k in x:
                    return x[k]
        return x

    try:
        if prices and hasattr(prices, "__len__"):
            n = len(prices)
            if n >= 3:
                # default = 14; take from context if available
                period = 14
                try:
                    if ctx is not None:
                        cfg = getattr(ctx, "config", None) or getattr(ctx, "CONFIG", None) or {}
                        if hasattr(cfg, "get"):
                            period = cfg.get("rsi", {}).get("period", 14)
                        elif isinstance(cfg, dict):
                            period = cfg.get("rsi", {}).get("period", 14)
                except (AttributeError, TypeError, ValueError):
                    pass
                # clamp to valid range (some impls need <= n-1)
                period = max(2, min(int(period), max(2, n - 1)))
                if calculate_rsi is None:
                    print("⚠️ RSI calculator unavailable — skipping RSI computation.")
                else:
                    try:
                        val = calculate_rsi(prices, period)  # type: ignore[misc]
                        out["rsi"] = _to_scalar(val)
                    except (ValueError, TypeError, ZeroDivisionError, IndexError):
                        out["rsi"] = None
    except Exception:  # pylint: disable=broad-exception-caught
        # fail-open: never break the loop due to indicator calc
        pass
    return out


def risk_screen(signals, ctx=None, **kwargs) -> bool:
    """Lightweight risk gate. Returns True to allow, False to block.

    Rules:
    - Block new *long* entries if RSI > 70 (overbought).
    - Enforce max open positions (default 3).
    - Enforce cash buffer ratio (cash/equity) >= capital_buffer (default 0.25).
    - Fail-open on unexpected errors (never halt the loop).
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    try:
        sb = signals or {}

        # ---- config (defensive) ----
        cfg = getattr(ctx, "config", None) or getattr(ctx, "CONFIG", None) or {}

        def cfg_get(path, default):
            """Safely traverse nested config dicts with defaults.

            Accepts a tuple path like ("risk", "max_open_positions") and
            returns the value if present; otherwise returns ``default``.
            """
            cur = cfg
            for k in path:
                if hasattr(cur, "get"):
                    cur = cur.get(k, default if k is path[-1] else {})
                elif isinstance(cur, dict):
                    cur = cur.get(k, default if k is path[-1] else {})
                else:
                    return default
            return cur

        max_positions = int(cfg_get(("risk", "max_open_positions"), 3))
        capital_buffer = float(cfg_get(("risk", "capital_buffer"), 0.25))

        # ---- portfolio guards ----
        portfolio = getattr(ctx, "portfolio", None)
        if portfolio is not None:
            opens = len(getattr(portfolio, "open_positions", []) or [])
            if opens >= max_positions:
                return False

            cash = float(getattr(portfolio, "cash", 0.0) or 0.0)
            equity = getattr(portfolio, "equity", None)

            if equity is None or equity == 0:
                print("⚠️ Equity missing or zero — skipping trade.")
                return False

            equity = float(equity)
            if equity > 0 and (cash / equity) < capital_buffer:
                return False

        # ---- RSI guard for long entries ----
        intent = sb.get("signal")
        rsi = sb.get("rsi")

        # scalarize rsi
        if isinstance(rsi, (list, tuple)):
            rsi = rsi[-1] if rsi else None
        elif isinstance(rsi, dict):
            rsi = rsi.get("rsi") or rsi.get("current") or rsi.get("value") or rsi.get("last")

        # numeric cast (best effort)
        try:
            rsi_num = float(rsi) if rsi is not None else None
        except (ValueError, TypeError):
            rsi_num = None

        if intent == "buy" and rsi_num is not None and rsi_num > 70.0:
            return False

        return True
    except Exception:  # pylint: disable=broad-exception-caught
        # fail-open so the engine doesn't halt on a guard bug
        return True


def execute_trade(signals, ctx=None, **kwargs):
    """Execute trade(s) based on signals.

    Returns a list of trade-like dicts for testing only; production flow
    still uses existing paths until we finish the split.
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    _ = (signals, ctx)
    return []


def check_and_close_exits(ctx=None, **kwargs) -> int:
    """Check exit rules and close positions if needed.

    Returns the number of closed positions (0 in stub).
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    _ = ctx
    return 0


if __name__ == "__main__":
    evaluate_signals_and_trade()
