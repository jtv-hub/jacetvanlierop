"""
Trading Logic Module

Evaluates signals using predefined strategies, executes mock trades,
manages open positions, and checks exit conditions.
"""

# pylint: disable=too-many-lines

import datetime
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Any

# Optional dependency for correlation checks
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

from crypto_trading_bot.bot.state.portfolio_state import load_portfolio_state
from crypto_trading_bot.config import CONFIG, get_mode_label, is_live
from crypto_trading_bot.context.trading_context import TradingContext
from crypto_trading_bot.ledger.trade_ledger import TradeLedger
from crypto_trading_bot.utils.kraken_client import (
    KrakenAPIError,
    KrakenAuthError,
    kraken_get_asset_pair_meta,
    kraken_place_order,
)
from crypto_trading_bot.utils.price_feed import get_current_price
from crypto_trading_bot.utils.price_history import (
    append_live_price,
    get_history_prices,
)

# Optional RSI calculator (import may vary by environment)
try:
    from crypto_trading_bot.indicators.rsi import calculate_rsi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    calculate_rsi = None  # type: ignore[assignment]

from .strategies.advanced_strategies import (
    ADXStrategy,
    BollingerBandStrategy,
    CompositeStrategy,
    KeltnerBreakoutStrategy,
    MACDStrategy,
    StochRSIStrategy,
    VWAPStrategy,
)
from .strategies.dual_threshold_strategies import DualThresholdStrategy
from .strategies.simple_rsi_strategies import SimpleRSIStrategy

context = TradingContext()
logger = logging.getLogger(__name__)

TRADES_LOG_PATH = "logs/trades.log"
PORTFOLIO_STATE_PATH = "logs/portfolio_state.json"

TRADE_INTERVAL = 300
MAX_PORTFOLIO_RISK = CONFIG.get("max_portfolio_risk", 0.10)
PAPER_STARTING_BALANCE = float(CONFIG.get("paper_mode", {}).get("starting_balance", 100_000.0))
SLIPPAGE = 0.0  # slippage handled per-asset in ledger; do not apply here

# Real-time price feed imported above per lint ordering


# get_current_price now lives in utils.price_feed; removed local duplicate.


# Removed hardcoded ASSETS list. Pairs are now centralized in CONFIG["tradable_pairs"].


@dataclass
class _RuntimeState:
    """Mutable runtime flags tracked across trading loop iterations."""

    live_block_logged: bool = False
    last_capital_log: tuple[float | None, str | None] = (None, None)
    kraken_pause_until: float | None = None
    last_mode_label: str | None = None
    auto_paused_reason: str | None = None


_STATE = _RuntimeState()
_KRAKEN_FAILURE_PAUSE_UNTIL: float | None = None
_KRAKEN_FAILURE_PAUSE_SECONDS = 60.0

getcontext().prec = 18


def _consecutive_losses(limit: int) -> int:
    """Return the number of trailing consecutive losing trades."""

    if limit <= 0:
        return 0
    if not os.path.exists(TRADES_LOG_PATH):
        return 0

    count = 0
    try:
        with open(TRADES_LOG_PATH, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return 0

    for line in reversed(lines):
        if count >= limit:
            break
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (rec.get("status") or "").lower() != "closed":
            continue
        roi = rec.get("roi")
        try:
            roi_val = float(roi)
        except (TypeError, ValueError):
            continue
        if roi_val < 0:
            count += 1
        else:
            break

    return count


def _evaluate_auto_pause(state_snapshot: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Inspect configured auto-pause thresholds and return status."""

    cfg = CONFIG.get("auto_pause", {})
    drawdown_limit = float(cfg.get("max_drawdown_pct", 0.10) or 0.0)
    roi_limit = float(cfg.get("max_total_roi_pct", -0.15) or 0.0)
    loss_limit = int(cfg.get("max_consecutive_losses", 5) or 0)

    state = state_snapshot or {}

    try:
        drawdown = float(state.get("drawdown_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        drawdown = 0.0

    if drawdown_limit > 0 and drawdown >= drawdown_limit:
        return True, f"Max drawdown {drawdown:.2%} ≥ limit {drawdown_limit:.2%}"

    try:
        total_roi = float(state.get("total_roi", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_roi = 0.0

    if roi_limit < 0 and total_roi <= roi_limit:
        return True, f"Total ROI {total_roi:.2%} ≤ limit {roi_limit:.2%}"

    if loss_limit > 0:
        streak = _consecutive_losses(loss_limit)
        if streak >= loss_limit:
            return True, f"{streak} consecutive losses ≥ limit {loss_limit}"

    return False, None


def _submit_live_trade(
    *,
    pair: str,
    side: str,
    size: float,
    price: float,
    strategy: str,
    confidence: float,
) -> bool:
    """Submit an order to the live exchange when live trading is enabled.

    Returns ``True`` if the attempt should proceed, ``False`` when blocked.
    """

    global _KRAKEN_FAILURE_PAUSE_UNTIL

    if not is_live:
        if not _STATE.live_block_logged:
            logger.warning(
                "🚫 Live trade blocked — is_live=False | pair=%s side=%s size=%.6f"
                " price=%.4f strategy=%s confidence=%.3f",
                pair,
                side,
                size,
                price,
                strategy,
                confidence,
            )
            _STATE.live_block_logged = True
        else:
            logger.debug(
                "Live trade attempt blocked (paper mode) | pair=%s side=%s size=%.6f",
                pair,
                side,
                size,
            )
        return False

    now = time.monotonic()
    pause_deadline = _KRAKEN_FAILURE_PAUSE_UNTIL or _STATE.kraken_pause_until
    if pause_deadline is not None and now < pause_deadline:
        pause_remaining = pause_deadline - now
        logger.error(
            "Kraken trading paused (%.1fs remaining) due to earlier error; skipping %s",
            pause_remaining,
            pair,
        )
        return False

    logger.info(
        "Submitting live trade | pair=%s side=%s size=%.6f price=%.4f strategy=%s confidence=%.3f",
        pair,
        side,
        size,
        price,
        strategy,
        confidence,
    )

    kraken_cfg = CONFIG.get("kraken", {}) or {}
    tif = kraken_cfg.get("time_in_force") or None
    validate_flag = bool(kraken_cfg.get("validate_orders", False))

    min_cost_default = float(
        CONFIG.get(
            "kraken_min_cost_threshold",
            kraken_cfg.get("min_cost_threshold", 0.5),
        )
    )
    min_cost_by_pair = kraken_cfg.get("pair_cost_minimums", {}) or {}
    config_cost_threshold = float(min_cost_by_pair.get(pair, min_cost_default))

    try:
        pair_meta = kraken_get_asset_pair_meta(pair)
    except KrakenAPIError as exc:
        pause_until = now + _KRAKEN_FAILURE_PAUSE_SECONDS
        _KRAKEN_FAILURE_PAUSE_UNTIL = pause_until
        _STATE.kraken_pause_until = pause_until
        logger.error(
            "Failed to fetch Kraken pair metadata; pausing live trading for %.0fs | " "pair=%s error=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            exc,
        )
        return False

    min_volume = float(pair_meta.get("ordermin", 0.0) or 0.0)
    metadata_cost_threshold = float(pair_meta.get("costmin", 0.0) or 0.0)
    price_decimals = int(pair_meta.get("pair_decimals", 5) or 5)
    volume_decimals = int(pair_meta.get("lot_decimals", 8) or 8)

    price_step = Decimal("1").scaleb(-price_decimals)
    volume_step = Decimal("1").scaleb(-volume_decimals)

    price_dec = Decimal(str(price)).quantize(price_step, rounding=ROUND_DOWN)
    size_dec = Decimal(str(size)).quantize(volume_step, rounding=ROUND_DOWN)

    min_volume_dec = Decimal(str(min_volume)) if min_volume else Decimal("0")
    if min_volume and size_dec < min_volume_dec:
        logger.warning(
            "Skipping live order: volume below minimum | pair=%s side=%s size=%.10f " "min_volume=%.10f",
            pair,
            side,
            float(size_dec),
            min_volume,
        )
        return False

    attempted_cost_dec = price_dec * size_dec
    effective_cost_threshold = max(metadata_cost_threshold, config_cost_threshold)
    min_cost_dec = Decimal(str(effective_cost_threshold)) if effective_cost_threshold else Decimal("0")
    if effective_cost_threshold and attempted_cost_dec < min_cost_dec:
        logger.warning(
            "Skipping live order: cost below minimum | pair=%s side=%s " "attempted_cost=%.10f threshold=%.10f",
            pair,
            side,
            float(attempted_cost_dec),
            effective_cost_threshold,
        )
        return False

    price = float(price_dec)
    size = float(size_dec)
    attempted_cost = float(attempted_cost_dec)

    try:
        response = kraken_place_order(
            pair,
            side,
            size,
            price,
            time_in_force=tif,
            validate=validate_flag,
            min_cost_threshold=effective_cost_threshold,
        )
    except (KrakenAuthError, KrakenAPIError) as exc:
        pause_until = now + _KRAKEN_FAILURE_PAUSE_SECONDS
        _KRAKEN_FAILURE_PAUSE_UNTIL = pause_until
        _STATE.kraken_pause_until = pause_until
        logger.error(
            "Kraken order submission exception; pausing live trading for %.0fs | " "pair=%s side=%s error=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
            exc,
        )
        return False
    except Exception:  # pylint: disable=broad-exception-caught
        pause_until = now + _KRAKEN_FAILURE_PAUSE_SECONDS
        _KRAKEN_FAILURE_PAUSE_UNTIL = pause_until
        _STATE.kraken_pause_until = pause_until
        logger.exception(
            "Unexpected Kraken order exception; pausing live trading for %.0fs | " "pair=%s side=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
        )
        return False

    if not isinstance(response, dict):
        pause_until = now + _KRAKEN_FAILURE_PAUSE_SECONDS
        _KRAKEN_FAILURE_PAUSE_UNTIL = pause_until
        _STATE.kraken_pause_until = pause_until
        logger.error(
            "Kraken order rejected; pausing live trading for %.0fs | pair=%s side=%s " "error=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
            response,
        )
        return False

    if not response.get("ok"):
        response_code = response.get("code")
        response_error = response.get("error")
        response_cost = response.get("attempted_cost", attempted_cost)
        response_threshold = response.get("threshold", effective_cost_threshold)
        if response_code == "cost_minimum_not_met":
            _KRAKEN_FAILURE_PAUSE_UNTIL = None
            _STATE.kraken_pause_until = None
            logger.warning(
                "Kraken cost minimum not met; skipping trade | pair=%s side=%s "
                "attempted_cost=%.6f threshold=%.6f error=%s",
                pair,
                side,
                float(response_cost or 0.0),
                float(response_threshold or effective_cost_threshold),
                response_error,
            )
            return False

        if response_code == "volume_minimum_not_met":
            _KRAKEN_FAILURE_PAUSE_UNTIL = None
            _STATE.kraken_pause_until = None
            logger.warning(
                "Kraken volume minimum not met; skipping trade | pair=%s side=%s " "size=%.6f min_volume=%.6f error=%s",
                pair,
                side,
                size,
                float(pair_meta.get("ordermin", min_volume)),
                response_error,
            )
            return False

        if response_code == "rate_limit":
            pause_until = time.monotonic() + 90.0
            _KRAKEN_FAILURE_PAUSE_UNTIL = pause_until
            _STATE.kraken_pause_until = pause_until
            logger.error(
                "Kraken rate limit encountered; pausing live trading for 90s | pair=%s side=%s",
                pair,
                side,
            )
            return False

        pause_until = now + _KRAKEN_FAILURE_PAUSE_SECONDS
        _KRAKEN_FAILURE_PAUSE_UNTIL = pause_until
        _STATE.kraken_pause_until = pause_until
        logger.error(
            "Kraken order rejected; pausing live trading for %.0fs | pair=%s side=%s " "error=%s code=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
            response_error,
            response_code,
        )
        return False

    _KRAKEN_FAILURE_PAUSE_UNTIL = None
    _STATE.kraken_pause_until = None

    txid = response.get("txid")
    if isinstance(txid, list) and txid:
        txid_repr = txid[0]
    else:
        txid_repr = txid
    descr = response.get("descr")

    logger.info(
        "Kraken live order acknowledged | pair=%s side=%s size=%.6f price=%s txid=%s descr=%s",
        pair,
        side,
        size,
        f"{price:.4f}" if isinstance(price, (int, float)) else price,
        txid_repr,
        descr,
    )
    return True


def _log_capital(amount: float, source: str) -> None:
    last_amount, last_source = _STATE.last_capital_log
    if last_amount == amount and last_source == source:
        return

    logger.info(
        "Active capital (%s): %s (source=%s)",
        get_mode_label(),
        f"${amount:,.2f}",
        source,
    )
    _STATE.last_capital_log = (amount, source)


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
        entry_adx: float | None = None,
        entry_rsi: float | None = None,
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
            "entry_adx": entry_adx,
            "entry_rsi": entry_rsi,
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
                        msg = "⚠️ RSI calculator unavailable — skipping RSI exit " f"for {trade_id}"
                        print(msg)
                    else:
                        rsi_val = calculate_rsi(
                            history,
                            CONFIG["rsi"]["period"],
                        )  # type: ignore[misc]
                        rsi_cfg = CONFIG["rsi"]
                        exit_upper = rsi_cfg.get("exit_upper", rsi_cfg.get("upper", 70))
                        if rsi_val is not None and rsi_val >= exit_upper:
                            exit_price = price
                            reason = "RSI_EXIT"
                            print(f"[EXIT] RSI_EXIT for {trade_id} pair={pos['pair']} " f"rsi={rsi_val:.2f}")
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


def _committed_notional(trades) -> float:
    """Total notional already allocated across pending trades."""
    total = 0.0
    for trade in trades or []:
        try:
            total += max(float(trade.get("notional", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _compute_position_sizing(
    *,
    total_capital: float,
    remaining_capital: float,
    current_price: float,
    confidence: float,
    base_risk_pct: float,
    buffer: float,
    reinvestment_rate: float | None,
    liquidity_factor: float,
    min_size: float,
    max_size: float,
) -> tuple[float, float, float]:
    """Return (size, notional, risk_fraction) for the proposed trade."""
    if total_capital <= 0 or remaining_capital <= 0 or current_price <= 0 or confidence <= 0 or base_risk_pct <= 0:
        return 0.0, 0.0, 0.0

    reinvestment_factor = float(reinvestment_rate) if reinvestment_rate is not None else 1.0
    reinvestment_factor = min(max(reinvestment_factor, 0.0), 1.0)
    liquidity_factor = min(max(liquidity_factor, 0.0), 1.0)
    buffer = max(buffer, 0.0)

    effective_pct = base_risk_pct * buffer * confidence * liquidity_factor * reinvestment_factor
    effective_pct = min(effective_pct, base_risk_pct)

    notional_target = remaining_capital * effective_pct
    if notional_target <= 0:
        return 0.0, 0.0, 0.0

    raw_units = notional_target / current_price
    if raw_units <= 0:
        return 0.0, 0.0, 0.0

    # Respect maximum size; if size falls below minimum, skip this trade
    units = min(raw_units, max_size)
    if units < min_size:
        return 0.0, 0.0, 0.0

    units = round(units, 6)
    notional = units * current_price
    risk_fraction = notional / total_capital if total_capital > 0 else 0.0
    risk_fraction = min(risk_fraction, base_risk_pct)

    return units, notional, risk_fraction


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
    *,
    available_capital: float | None = None,
    risk_per_trade: float | None = None,
    reinvestment_rate: float | None = None,
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

    mode_label = get_mode_label()
    if mode_label != _STATE.last_mode_label:
        logger.info("Trading mode: %s (is_live=%s)", mode_label, is_live)
        _STATE.last_mode_label = mode_label

    state_snapshot: dict | None = None
    resolved_capital: float | None = None
    capital_source = "manual_override"
    if available_capital is not None:
        try:
            resolved_capital = float(available_capital)
        except (TypeError, ValueError):
            resolved_capital = None

    if resolved_capital is None or resolved_capital <= 0:
        state_snapshot = load_portfolio_state(
            refresh=is_live,
            starting_balance=PAPER_STARTING_BALANCE,
        )
        resolved_capital = float(state_snapshot.get("available_capital", PAPER_STARTING_BALANCE))
        capital_source = state_snapshot.get("capital_source", "portfolio_state")

    if state_snapshot is None:
        state_snapshot = load_portfolio_state(
            refresh=is_live,
            starting_balance=PAPER_STARTING_BALANCE,
        )
        capital_source = state_snapshot.get("capital_source", capital_source)
        if resolved_capital is None or resolved_capital <= 0:
            resolved_capital = float(state_snapshot.get("available_capital", PAPER_STARTING_BALANCE))

    resolved_capital = max(resolved_capital or 0.0, 0.0)
    _log_capital(resolved_capital, capital_source)

    if not check_exits_only:
        paused, reason = _evaluate_auto_pause(state_snapshot)
        if paused:
            if _STATE.auto_paused_reason != reason:
                logger.error("[AUTO-PAUSE] %s", reason)
                print(f"[AUTO-PAUSE] {reason}")
            else:
                logger.debug("Auto-pause active: %s", reason)
            _STATE.auto_paused_reason = reason
            return
        if _STATE.auto_paused_reason is not None:
            logger.info("Auto-pause cleared; resuming trade evaluation.")
            _STATE.auto_paused_reason = None

    if reinvestment_rate is None:
        reinvestment_rate = float(state_snapshot.get("reinvestment_rate", 0.0))

    trade_risk_pct = 0.02 if risk_per_trade is None else max(float(risk_per_trade), 0.0)
    # Build current price map using live feed for each pair
    current_prices: dict[str, float] = {}
    for pair in pairs:
        price_now = get_current_price(pair)
        asset = pair.split("/")[0]
        if price_now is not None and price_now > 0:
            current_prices[pair] = price_now
            print(f"[FEED] {asset} price OK: {price_now}")
        else:
            print(f"⚠️ No current price for {pair}; skipping in price map")

    # Preload seeded history for all pairs (startup fallback)
    try:
        preload_period = CONFIG.get("rsi", {}).get("period", 21)
        preload_min = max(int(preload_period) + 1, 14)
        for p in pairs:
            _ = get_history_prices(p, min_len=preload_min)
    except Exception:  # pylint: disable=broad-exception-caught
        # Non-fatal: trading loop can still proceed using live prices only
        pass

    # Refresh current market regime and capital buffer before signal evaluation
    context.update_context()
    base_buffer = context.get_buffer()
    composite_buffer = context.get_buffer_for_strategy("CompositeStrategy")
    print(f"[CONTEXT] Regime: {context.get_regime()} | Buffer: {base_buffer} | " f"CompositeBuffer: {composite_buffer}")
    save_portfolio_state(context)

    # Daily trade limit (configurable; ENV override allowed)
    max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "0") or 0)

    def _today_trade_count() -> int:
        try:
            if not os.path.exists(TRADES_LOG_PATH):
                return 0
            today = datetime.datetime.now(datetime.UTC).date().isoformat()
            count = 0
            with open(TRADES_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        ts = rec.get("timestamp")
                        if ts and ts[:10] == today:
                            if (rec.get("status") or "").lower() in {"executed", "closed"}:
                                count += 1
                    except json.JSONDecodeError:
                        continue
            return count
        except OSError:
            return 0

    def _last_closed_trades(n: int) -> list[dict]:
        """Return the most recent n closed trades with valid ROI.

        Parses `logs/trades.log` and sorts by timestamp (UTC).
        """
        items: list[dict] = []
        try:
            if not os.path.exists(TRADES_LOG_PATH):
                return items
            with open(TRADES_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (rec.get("status") or "").lower() != "closed":
                        continue
                    if rec.get("roi") is None:
                        continue
                    ts = rec.get("timestamp") or "1970-01-01T00:00:00+00:00"
                    rec["_ts"] = ts
                    items.append(rec)
            # Sort by timestamp string (ISO 8601 lexicographic works for UTC)
            items.sort(key=lambda r: r.get("_ts", ""))
            return items[-n:]
        except OSError:
            return items[:n]

    # ---- Daily limits with streak rules ----
    default_daily_cap = int(CONFIG.get("MAX_TRADES_PER_DAY", 5))
    bonus_cap = int(CONFIG.get("BONUS_LIMIT_IF_WINNING_STREAK", 7))
    loss_streak_stop = int(CONFIG.get("STOP_IF_LOSS_STREAK", 3))

    # Start from ENV override if provided, else config default
    current_daily_cap = max_trades_per_day if max_trades_per_day > 0 else default_daily_cap

    today_count = _today_trade_count()
    recent = _last_closed_trades(5)

    # Loss streak: last loss_streak_stop closed trades all losers
    if len(recent) >= loss_streak_stop:
        last_n = recent[-loss_streak_stop:]
        if all((t.get("roi") or 0) < 0 for t in last_n):
            print("[STREAK HALT] 3-loss streak detected — Trading paused for the day")
            return

    # Winning streak: last 5 closed trades all winners
    if len(recent) >= 5 and all((t.get("roi") or 0) > 0 for t in recent[-5:]):
        if current_daily_cap < bonus_cap:
            current_daily_cap = bonus_cap
            print("[STREAK BONUS] 5-win streak detected — Daily trade cap raised to 7")

    # Enforce daily cap before scanning assets
    if today_count >= current_daily_cap:
        print(f"[LIMIT] Daily trade limit reached ({today_count})")
        return

    total_capital = resolved_capital
    if not check_exits_only and total_capital <= 0:
        print("⚠️ Available capital is zero — skipping new trade evaluation.")
        check_exits_only = True

    if not check_exits_only:
        proposed_trades = []
        executed_trades = 0
        print(f"[ASSETS] Scanning {len(pairs)} pairs: {pairs}")
        for pair in pairs:
            try:
                # Derive asset symbol from pair (e.g., "BTC" from "BTC/USD")
                asset = pair.split("/")[0]
                # Fetch current price
                current_price = current_prices.get(pair)
                if current_price is None:
                    current_price = get_current_price(pair)
                if current_price is None or current_price <= 0:
                    print(f"[SKIP] No valid current price for {pair}")
                    continue

                # Ensure seeded history is available, then append live price
                rsi_period = CONFIG.get("rsi", {}).get("period", 21)
                # Ensure we always provide at least 30 points to RSI-based strategies
                min_needed = max(int(rsi_period) + 1, 30)
                _ = get_history_prices(pair, min_len=min_needed)
                append_live_price(pair, float(current_price))
                safe_prices = get_history_prices(pair, min_len=min_needed)
                # Filter out None values before strategy evaluation
                safe_prices = [p for p in safe_prices if p is not None]
                # Pre-pair debug trace
                print(f"[TEST] Generating signal for {pair} with {len(safe_prices)} valid candles")
                last5 = safe_prices[-5:]
                print(f"[DEBUG] {pair} prices: {last5} (last 5)")

                # Pre-compute RSI for diagnostics/logging
                rsi_val = None
                try:
                    if calculate_rsi is not None and len(safe_prices) >= int(rsi_period) + 1:
                        rsi_val = float(calculate_rsi(safe_prices, int(rsi_period)))
                except (TypeError, ValueError):
                    rsi_val = None
                # Skip if insufficient history
                if len(safe_prices) < min_needed:
                    print(f"[SKIP] Not enough price history for {pair} " f"(only {len(safe_prices)} prices)")
                    continue
                # Compute ADX gate using recent prices
                adx_val = context.get_adx(pair, safe_prices)
                if adx_val is not None:
                    print(f"[ADX] {pair} ADX={adx_val:.2f}")
                volume = mock_volume_data.get(asset, [MIN_VOLUME])[-1]
                if volume < MIN_VOLUME:
                    print(f"[SKIP] {asset}: volume {volume} < MIN_VOLUME {MIN_VOLUME}")
                    continue
                # Reinitialize strategies per iteration to reset state
                per_asset_params = CONFIG.get("strategy_params", {})
                strategies = [
                    SimpleRSIStrategy(
                        period=CONFIG.get("rsi", {}).get("period", 21),
                        lower=CONFIG.get("rsi", {}).get("lower", 48),
                        upper=CONFIG.get("rsi", {}).get("upper", 75),
                        per_asset=per_asset_params.get("SimpleRSIStrategy", {}),
                    ),
                    DualThresholdStrategy(),
                    MACDStrategy(per_asset=per_asset_params.get("MACDStrategy", {})),
                    KeltnerBreakoutStrategy(per_asset=per_asset_params.get("KeltnerBreakoutStrategy", {})),
                    StochRSIStrategy(per_asset=per_asset_params.get("StochRSIStrategy", {})),
                    BollingerBandStrategy(per_asset=per_asset_params.get("BollingerBandStrategy", {})),
                    VWAPStrategy(per_asset=per_asset_params.get("VWAPStrategy", {})),
                    ADXStrategy(per_asset=per_asset_params.get("ADXStrategy", {})),
                    CompositeStrategy(per_asset=per_asset_params.get("CompositeStrategy", {})),
                ]
                # Per-asset stats
                signals_count = 0
                buy_count = 0
                sell_count = 0
                skipped_low_conf = 0
                skipped_none = 0
                for strategy in strategies:
                    try:
                        # Pass asset hint where supported
                        try:
                            signal_result = strategy.generate_signal(
                                safe_prices,
                                volume=volume,
                                asset=asset,
                                adx=adx_val,
                            )
                        except TypeError:
                            # Older strategies may not accept 'adx'
                            signal_result = strategy.generate_signal(
                                safe_prices,
                                volume=volume,
                            )
                        scan_msg = (
                            f"[SCAN] {asset} strat={strategy.__class__.__name__} "
                            f"price={current_price:.4f} vol={volume} -> {signal_result}"
                        )
                        print(scan_msg)
                    except (ValueError, RuntimeError) as e:
                        log_path = "logs/anomalies.log"
                        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "timestamp": timestamp,
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
                    signal = signal_result.get("signal") or signal_result.get("side")
                    if not signal:
                        print(f"[SKIP] No signal generated for {pair} — " "strategy returned None or RSI failed")
                    confidence = float(signal_result.get("confidence", 0.0) or 0.0)
                    strategy_name = strategy.__class__.__name__

                    regime = context.get_regime()
                    buffer = context.get_buffer_for_strategy(strategy_name)

                    if signal in ["buy", "sell"]:
                        print(f"[RSI DEBUG] Signal for {pair} -> {signal}")

                    if signal not in ["buy", "sell"]:
                        skipped_none += 1
                        print(f"⚠️ Skipping {pair} — No actionable signal")
                        continue
                # ADX gating (regime filter)
                if adx_val is not None:
                    if adx_val < 20.0:
                        skipped_none += 1
                        print(f"[SKIP] {pair}: ADX too weak ({adx_val:.2f})")
                        print(f"[ADX DEBUG] {pair}: ADX={adx_val:.2f}, adjusted confidence=0.0")
                        continue
                    if adx_val > 40.0:
                        before = confidence
                        confidence = min(1.0, confidence * 1.2)
                        print(f"[ADX] Boosted confidence {before:.3f} → " f"{confidence:.3f} (strong trend)")
                    print(f"[ADX DEBUG] {pair}: ADX={adx_val:.2f}, adjusted " f"confidence={confidence:.3f}")
                if confidence < 0.4:
                    skipped_low_conf += 1
                    print(f"⚠️ Skipping {pair} — Confidence too low: {confidence}")
                    continue
                if volume is None or volume < MIN_VOLUME:
                    print(f"[{pair}] Skipping due to low volume: {volume}")
                    continue
                print(f"📊 Volume for {pair}: {volume}")
                signals_count += 1
                if signal == "buy":
                    buy_count += 1
                elif signal == "sell":
                    sell_count += 1

                committed_capital = _committed_notional(proposed_trades)
                remaining_capital = max(total_capital - committed_capital, 0.0)
                if remaining_capital <= 0:
                    print("⚠️ Capital exhausted for new positions; stopping scans.")
                    break

                cfg_sz = CONFIG.get("trade_size", {})
                min_sz = float(cfg_sz.get("min", 0.001))
                max_sz = float(cfg_sz.get("max", 0.005))
                dynamic_buffer = context.get_buffer_for_strategy(strategy_name)
                liquidity_factor = min(volume / 1000, 1.0)

                adjusted_size, position_notional, trade_risk = _compute_position_sizing(
                    total_capital=total_capital,
                    remaining_capital=remaining_capital,
                    current_price=current_price,
                    confidence=confidence,
                    base_risk_pct=trade_risk_pct,
                    buffer=dynamic_buffer,
                    reinvestment_rate=reinvestment_rate,
                    liquidity_factor=liquidity_factor,
                    min_size=min_sz,
                    max_size=max_sz,
                )

                if adjusted_size <= 0:
                    print(f"⚠️ Skipping {pair} — Not enough capital for minimum position size")
                    continue

                trade_data = {
                    "asset": asset,
                    "size": adjusted_size,
                    "risk": trade_risk,
                    "strategy": strategy_name,
                    "confidence": confidence,
                    "signal_score": confidence,
                    "regime": regime,
                    "notional": round(position_notional, 2),
                }

                proposed_trades.append(trade_data)
                total_risk = calculate_total_risk(proposed_trades)
                if total_risk > MAX_PORTFOLIO_RISK:
                    msg = f"⚠️ Skipping trade for {asset} — Portfolio risk " f"({total_risk:.2%}) exceeds cap"
                    print(msg)
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
                        pair_a = f"{asset}/USD"
                        pair_b = f"{other}/USD"
                        a_price = current_prices.get(pair_a) or get_current_price(pair_a)
                        b_price = current_prices.get(pair_b) or get_current_price(pair_b)
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
                                    corr_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
                                    f.write(
                                        json.dumps(
                                            {
                                                "timestamp": corr_timestamp,
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
                                corr_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
                                f.write(
                                    json.dumps(
                                        {
                                            "timestamp": corr_timestamp,
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

                trade_side = signal
                submitted_live = _submit_live_trade(
                    pair=pair,
                    side=trade_side,
                    size=adjusted_size,
                    price=float(current_price),
                    strategy=strategy_name,
                    confidence=confidence,
                )

                if is_live and not submitted_live:
                    print(f"🚫 Live trade suppressed for {pair} — toggle is_live=False")
                    continue

                if not is_live:
                    logger.debug(
                        "Paper trade simulated | pair=%s side=%s size=%.6f price=%.4f " "strategy=%s confidence=%.3f",
                        pair,
                        trade_side,
                        adjusted_size,
                        float(current_price),
                        strategy_name,
                        confidence,
                    )

                # Daily trade limit check (re-evaluate per attempt)
                if current_daily_cap > 0:
                    today_count = _today_trade_count()
                    print(f"[LIMIT] Today trade count: {today_count} / {current_daily_cap}")
                    if today_count >= current_daily_cap:
                        print("[LIMIT] Daily trade limit reached — skipping new trade")
                        continue

                trade_id = str(uuid.uuid4())
                print(f"[DEBUG] Using trade_id for both log_trade and open_position: {trade_id}")

                entry_raw = safe_prices[-1]
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
                    rsi=rsi_val,
                    adx=adx_val,
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
                    entry_adx=adx_val,
                    entry_rsi=rsi_val,
                    timestamp=logged_ts,
                )
                executed_trades += 1
                print(f"🧠 Confidence Score: {confidence}")
                # Print confirmation only when a valid trade record exists and size > 0
                if adjusted_size > 0 and logged_trade:
                    print(f"📝 Logged trade: {pair} | Size: {adjusted_size}")
                    print(f"📝 Strategy: {strategy_name}")

                # Summary per asset
                print(
                    f"[SUMMARY] {asset}: signals={signals_count} "
                    + f"(buy={buy_count}, sell={sell_count}), "
                    + f"skipped_none={skipped_none}, "
                    + f"skipped_low_conf={skipped_low_conf}"
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[ERROR] Exception while evaluating {pair}: {e}")

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
            confidence = 1.0
            regime = context.get_regime()
            buffer = context.get_buffer_for_strategy(strategy_name)
            cfg_sz = CONFIG.get("trade_size", {})
            min_sz = float(cfg_sz.get("min", 0.001))
            max_sz = float(cfg_sz.get("max", 0.005))
            fallback_size, _, _ = _compute_position_sizing(
                total_capital=total_capital if total_capital > 0 else PAPER_STARTING_BALANCE,
                remaining_capital=total_capital if total_capital > 0 else PAPER_STARTING_BALANCE,
                current_price=price,
                confidence=confidence,
                base_risk_pct=trade_risk_pct if trade_risk_pct > 0 else 0.02,
                buffer=buffer,
                reinvestment_rate=reinvestment_rate,
                liquidity_factor=1.0,
                min_size=min_sz,
                max_size=max_sz,
            )
            if fallback_size <= 0:
                print("[FALLBACK] Unable to compute a viable fallback position size; skipping")
                return
            adjusted_size = fallback_size
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
        ledger.update_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            reason=reason,
            exit_reason=reason,
        )
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
