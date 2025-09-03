"""
Trade Ledger Module (Class-Based)

Encapsulates trade lifecycle logging, ROI updates, schema validation,
and trade syncing using a class-based approach.
"""

import fcntl
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone

from crypto_trading_bot.bot.utils.schema_validator import validate_trade_schema
from crypto_trading_bot.config import CONFIG

# Enable extra debug output when explicitly requested
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"

TRADES_LOG_PATH = "logs/trades.log"
SYSTEM_LOG_PATH = "logs/system.log"
POSITIONS_PATH = "logs/positions.jsonl"
SLIPPAGE = 0.002  # default fallback; per-asset slippage from CONFIG overrides this

# System logger (warnings/errors/debug) to logs/system.log
system_logger = logging.getLogger("trade_ledger.system")
if not system_logger.hasHandlers():
    os.makedirs("logs", exist_ok=True)
    sys_handler = logging.FileHandler(SYSTEM_LOG_PATH, encoding="utf-8")
    sys_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    system_logger.addHandler(sys_handler)
    system_logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    system_logger.propagate = False

# Trade logger (message-only JSON lines) to logs/trades.log
trade_logger = logging.getLogger("trade_ledger.trades")
if not trade_logger.hasHandlers():
    os.makedirs("logs", exist_ok=True)
    trade_handler = logging.FileHandler(TRADES_LOG_PATH, encoding="utf-8")
    trade_handler.setFormatter(logging.Formatter("%(message)s"))
    trade_logger.addHandler(trade_handler)
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False


def _slippage_rate_for_pair(pair: str) -> float:
    base = (pair or "BTC/USD").split("/")[0]
    majors = {"BTC", "ETH"}
    if base in majors:
        return float(CONFIG.get("slippage", {}).get("majors", 0.001))
    cfg = CONFIG.get("slippage", {})
    use_random = bool(cfg.get("use_random", False))
    lo = float(cfg.get("alts_min", 0.005))
    hi = float(cfg.get("alts_max", 0.01))
    if use_random:
        return random.uniform(lo, hi)
    return (lo + hi) / 2.0


def _apply_slippage(pair: str, price: float, side: str) -> tuple[float, float, float]:
    """Apply side-aware slippage using configured rates.

    Returns (adjusted_price, slippage_amount, slippage_rate).
    """
    rate = _slippage_rate_for_pair(pair)
    raw = float(price)
    if side == "buy":
        adjusted = raw * (1 + rate)
        amount = adjusted - raw
    else:  # sell
        adjusted = raw * (1 - rate)
        amount = raw - adjusted
    adjusted = round(adjusted, 4)
    amount = round(amount, 4)
    system_logger.debug(
        "Applied slippage pair=%s side=%s raw=%.6f adj=%.6f rate=%.6f amount=%.6f",
        pair,
        side,
        raw,
        adjusted,
        rate,
        amount,
    )
    return adjusted, amount, rate


def _apply_exit_slippage(pair: str, exit_price: float) -> float:
    """Backward-compatible no-op wrapper retained for tests.

    Exit slippage is applied via _apply_slippage in update paths. This function
    preserves previous external expectations that it returns the input value.
    """
    _ = pair
    return float(exit_price)


class TradeLedger:
    """
    A class to manage trade lifecycle logging, updating, and validation.
    Handles logging trade entries, updating trade exits, syncing positions,
    reloading historical trades, and verifying lifecycle status.
    """

    def __init__(self, position_manager):
        """
        Initialize the TradeLedger with a reference to the PositionManager.
        Loads existing trades from the trades log.
        """
        self.trades = []
        self.trade_index: dict[str, dict] = {}
        self.position_manager = position_manager
        self.reload_trades()

    def log_trade(self, trading_pair, trade_size, strategy_name, **kwargs):
        """
        Log a new trade to the trades log.
        Validates schema and writes the trade to disk.
        Returns the trade ID.
        """
        if not isinstance(strategy_name, str):
            raise ValueError(f"[Ledger] Invalid strategy_name: {strategy_name}")
        confidence = float(kwargs.get("confidence", 0.0) or 0.0)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"[Ledger] Invalid confidence value: {confidence}")
        if not isinstance(trade_size, (int, float)) or trade_size <= 0:
            raise ValueError(f"[Ledger] Invalid trade size: {trade_size}")
        # Bound-check size based on CONFIG
        trade_size = float(trade_size)
        min_sz = float(CONFIG.get("trade_size", {}).get("min", 0.0001))
        max_sz = float(CONFIG.get("trade_size", {}).get("max", max(trade_size, 1.0)))
        trade_size = max(min_sz, min(trade_size, max_sz))
        if "entry_price" in kwargs:
            entry_price_val = kwargs["entry_price"]
            if not isinstance(entry_price_val, (int, float)) or entry_price_val <= 0:
                raise ValueError(f"[Ledger] Invalid entry_price: {entry_price_val}")

        trade_id = kwargs.get("trade_id") or str(uuid.uuid4())
        entry_price = kwargs.get("entry_price", 18000 + 250 * (0.5 - random.random()))
        # Prefer explicit side; fallback to legacy inference to preserve behavior
        side = kwargs.get("side")
        if side not in ("buy", "sell"):
            side = "buy" if strategy_name.lower().find("sell") == -1 else "sell"
        entry_price_adj, slippage_amount_entry, slip_rate = _apply_slippage(
            trading_pair,
            float(entry_price),
            side,
        )

        capital_buffer = kwargs.get("capital_buffer", 0.25)
        tax_method = kwargs.get("tax_method", "FIFO")
        regime = kwargs.get("regime", "unknown")

        trade = {
            "trade_id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": trading_pair,
            "size": trade_size,
            "strategy": strategy_name,
            "confidence": confidence,
            "status": "executed",
            "capital_buffer": capital_buffer,
            "tax_method": tax_method,
            "cost_basis": round(entry_price_adj * trade_size, 4),
            "entry_price": entry_price_adj,
            # Persist side so downstream logic can apply correct exit behavior
            "side": side,
            "exit_price": None,
            "realized_gain": None,
            "holding_period_days": None,
            "roi": None,
            "reason": None,
            "regime": regime,
            # Slippage metadata
            "entry_slippage_rate": round(slip_rate, 6),
            "entry_slippage_amount": slippage_amount_entry,
        }

        # Debug: emit the trade being logged for diagnostics (opt-in)
        if DEBUG_MODE:
            try:
                system_logger.debug("Logging trade: %s", json.dumps(trade, indent=2))
            except (TypeError, OSError):
                pass

        # Validate schema; on failure, log anomaly for audit and re-raise.
        try:
            validate_trade_schema(trade)
        except (ValueError, TypeError) as e:
            try:
                os.makedirs("logs", exist_ok=True)
                with open("logs/anomalies.log", "a", encoding="utf-8") as af:
                    af.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "type": "Trade Schema Error",
                                "error": str(e),
                                "trade": trade,
                            }
                        )
                        + "\n"
                    )
            except (OSError, IOError):
                pass
            raise

        # Write compact, one-line JSON to trades.log via trade_logger
        trade_logger.info(json.dumps(trade, separators=(",", ":")))
        self.trades.append(trade)
        self.trade_index[trade_id] = trade
        return trade_id

    def open_position(self, trade_id, trade):
        """
        Record a newly opened trade into the positions file (positions.jsonl).
        Uses the trade ID as key reference.
        """
        trade["trade_id"] = trade_id
        with open(POSITIONS_PATH, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(trade, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
        system_logger.debug("Position %s written to positions.jsonl", trade_id)

    def update_trade(self, trade_id, exit_price, reason):
        """
        Update a trade with exit information such as exit_price, reason, and ROI.
        If trade is missing, attempts to reload from file and/or reconstruct from positions.
        Retries up to 3 times on failure and safely resyncs positions.jsonl.
        """
        if not isinstance(exit_price, (int, float)) or exit_price <= 0:
            raise ValueError(f"[Ledger] Invalid exit_price for trade {trade_id}: {exit_price}")

        if not os.path.exists(TRADES_LOG_PATH):
            system_logger.error("Trade file not found: %s", TRADES_LOG_PATH)
            return

        if not hasattr(self.position_manager, "positions") or self.position_manager.positions is None:
            system_logger.error("position_manager.positions unavailable for trade %s", trade_id)
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                updated = False

                # Ensure we have latest trades in memory
                trades = self.trades or []
                if trade_id not in self.trade_index:
                    trades = self.reload_trades()

                # If still not present, try reconstruct from positions
                if trade_id not in self.trade_index:
                    pos = self.position_manager.positions.get(trade_id)
                    if pos and pos.get("exit_price") is None:
                        trade = {
                            "trade_id": trade_id,
                            "timestamp": pos.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat(),
                            ),
                            "pair": pos.get("pair", "BTC/USD"),
                            "size": pos.get("size", 0.001),
                            "strategy": pos.get("strategy", "Unknown"),
                            "confidence": float(pos.get("confidence", 0.0) or 0.0),
                            "status": "executed",
                            "capital_buffer": 0.25,
                            "tax_method": "FIFO",
                            "cost_basis": round(
                                pos.get("entry_price", 18000) * pos.get("size", 0.001),
                                4,
                            ),
                            "entry_price": round(pos.get("entry_price", 18000), 4),
                            "exit_price": None,
                            "realized_gain": None,
                            "holding_period_days": None,
                            "roi": None,
                            "reason": None,
                            "regime": "unknown",
                        }
                        trades.append(trade)
                        self.trade_index[trade_id] = trade
                        system_logger.warning("Synced missing trade %s from position", trade_id)

                # If still not found, log a clear error and stop
                if trade_id not in self.trade_index:
                    system_logger.error(
                        "Trade ID %s not found in memory or trades.log — aborting update",
                        trade_id,
                    )
                    return

                # Apply update
                updated = False
                t_obj = self.trade_index.get(trade_id)
                if t_obj and t_obj.get("exit_price") is None:
                    size = t_obj.get("size")
                    entry_price = t_obj.get("entry_price")
                    # Derive entry_price if missing and cost_basis/size available
                    if entry_price is None and t_obj.get("cost_basis") is not None and size:
                        try:
                            entry_price = float(t_obj["cost_basis"]) / float(size)
                        except (TypeError, ZeroDivisionError):
                            entry_price = None
                    if entry_price is None:
                        system_logger.error(
                            "Missing entry_price for trade %s; cannot compute PnL/ROI",
                            trade_id,
                        )
                        return
                    try:
                        timestamp = datetime.fromisoformat(t_obj["timestamp"])
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError) as e:
                        system_logger.error(
                            "Invalid timestamp for trade %s: %s (error: %s)",
                            trade_id,
                            t_obj.get("timestamp"),
                            e,
                        )
                        return

                    # Use opposite side for exit to account for shorts
                    exit_side = "buy" if t_obj.get("side") == "sell" else "sell"
                    exit_adj, exit_slippage_amount, slip_rate = _apply_slippage(
                        t_obj.get("pair"), float(exit_price), exit_side
                    )
                    t_obj.update(
                        {
                            "exit_price": exit_adj,
                            "status": "closed",
                            "reason": reason,
                            "holding_period_days": round(
                                (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400,
                                2,
                            ),
                            "realized_gain": round((exit_adj - entry_price) * size, 4),
                            "roi": (round((exit_adj - entry_price) / entry_price, 6) if entry_price else 0.0),
                            "exit_slippage_rate": round(slip_rate, 6),
                            "exit_slippage_amount": exit_slippage_amount,
                        }
                    )
                    # Revalidate updated trade; log anomaly but do not abort update flow
                    try:
                        validate_trade_schema(t_obj)
                    except (ValueError, TypeError) as e:
                        try:
                            os.makedirs("logs", exist_ok=True)
                            with open("logs/anomalies.log", "a", encoding="utf-8") as af:
                                af.write(
                                    json.dumps(
                                        {
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "type": "Trade Schema Error (post-update)",
                                            "error": str(e),
                                            "trade": t_obj,
                                        }
                                    )
                                    + "\n"
                                )
                        except (OSError, IOError):
                            pass
                    updated = True

                if updated:
                    # Safely rewrite trades.log with exclusive lock held on source
                    temp_path = TRADES_LOG_PATH + ".tmp"
                    with open(TRADES_LOG_PATH, "r+", encoding="utf-8") as src:
                        fcntl.flock(src, fcntl.LOCK_EX)
                        try:
                            with open(temp_path, "w", encoding="utf-8") as f:
                                for t in trades:
                                    if t is not None:
                                        f.write(json.dumps(t) + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(temp_path, TRADES_LOG_PATH)
                        finally:
                            fcntl.flock(src, fcntl.LOCK_UN)
                    self.trades = trades
                    self.trade_index = {t.get("trade_id"): t for t in trades if t and t.get("trade_id")}
                    system_logger.debug("Successfully updated trade %s in trades.log", trade_id)

                    # Safely resync positions.jsonl by removing the closed position
                    try:
                        if os.path.exists(POSITIONS_PATH):
                            tmp_pos = POSITIONS_PATH + ".tmp"
                            with open(POSITIONS_PATH, "r", encoding="utf-8") as src:
                                with open(tmp_pos, "w", encoding="utf-8") as dst:
                                    fcntl.flock(dst, fcntl.LOCK_EX)
                                    try:
                                        for line in src:
                                            try:
                                                obj = json.loads(line)
                                                if obj.get("trade_id") == trade_id:
                                                    # skip removed position
                                                    continue
                                            except json.JSONDecodeError:
                                                # preserve unparseable lines as-is
                                                pass
                                            dst.write(line)
                                        dst.flush()
                                        os.fsync(dst.fileno())
                                    finally:
                                        fcntl.flock(dst, fcntl.LOCK_UN)
                            os.replace(tmp_pos, POSITIONS_PATH)
                            system_logger.debug(
                                "Synchronized positions.jsonl — removed closed position %s",
                                trade_id,
                            )
                    except (OSError, IOError) as e:
                        system_logger.warning("Failed to sync positions.jsonl for %s: %s", trade_id, e)
                else:
                    # If trade is already closed, treat this as idempotent
                    existing = next(
                        (t for t in trades if t.get("trade_id") == trade_id),
                        None,
                    )
                    if existing and existing.get("exit_price") is not None and existing.get("status") == "closed":
                        system_logger.info("Idempotent update — trade %s already closed", trade_id)
                    else:
                        system_logger.info("No update applied — trade_id %s not in open state", trade_id)
                return

            except (OSError, IOError, json.JSONDecodeError) as e:
                system_logger.error("Error updating trade (attempt %s): %s", attempt + 1, e)
                time.sleep(1)

        system_logger.error("Failed to update trade %s after %s attempts", trade_id, max_retries)

    def reload_trades(self):
        """
        Reload all trades from the trades log into memory.
        Skips malformed or incomplete entries.
        Returns the list of trades.
        """
        self.trades = []
        if os.path.exists(TRADES_LOG_PATH):
            with open(TRADES_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        trade = json.loads(line)
                        if trade.get("trade_id"):
                            self.trades.append(trade)
                    except json.JSONDecodeError as e:
                        raw = line.strip()
                        redacted = (raw[:200] + "...") if len(raw) > 200 else raw
                        system_logger.warning(
                            "Skipping malformed line in trades.log: %s — Error: %s",
                            redacted,
                            e,
                        )
            self.trade_index = {t.get("trade_id"): t for t in self.trades if t and t.get("trade_id")}
            system_logger.info("Reloaded %s trades from trades.log", len(self.trades))
        else:
            system_logger.info("No trades.log file found.")
        return self.trades

    def verify_trade_update(self, target_trade_id):
        """
        Verify if a trade with the given trade_id has been updated properly.
        Checks for required fields and status.
        Returns True if updated, False otherwise.
        """
        trades = self.trades
        matching = [t for t in trades if t.get("trade_id") == target_trade_id]
        if not matching:
            system_logger.info("No entries found for trade_id: %s", target_trade_id)
            return False
        latest = max(matching, key=lambda x: x.get("timestamp", "1970-01-01T00:00:00+00:00"))
        required_fields = ["status", "exit_price", "reason", "roi"]
        status_ok = latest.get("status") == "closed"
        fields_ok = all(latest.get(f) is not None for f in required_fields)
        if status_ok and fields_ok:
            system_logger.info("Verified update for trade %s", target_trade_id)
            return True
        system_logger.info("Trade found but not fully updated: %s", latest)
        return False
