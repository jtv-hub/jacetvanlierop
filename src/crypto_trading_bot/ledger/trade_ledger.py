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

from crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler
from crypto_trading_bot.bot.utils.schema_validator import validate_trade_schema
from crypto_trading_bot.config import CONFIG

logger = logging.getLogger("trade_ledger")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    rotating_handler = get_rotating_handler("trades.log")
    logger.addHandler(rotating_handler)

TRADES_LOG_PATH = "logs/trades.log"
POSITIONS_PATH = "logs/positions.jsonl"
SLIPPAGE = 0.002  # default fallback; per-asset slippage from CONFIG overrides this


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


def _apply_exit_slippage(pair: str, exit_price: float) -> float:
    """Stub slippage helper (no-op) to preserve existing behavior.

    Parameters
    ----------
    pair : str
        Trading pair, e.g., "BTC/USD".
    exit_price : float
        Raw exit price.

    Returns
    -------
    float
        Unmodified price (refactor hook point).
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
        confidence = kwargs.get("confidence", 0.0)
        if not isinstance(confidence, float) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"[Ledger] Invalid confidence value: {confidence}")
        if not isinstance(trade_size, (int, float)) or trade_size <= 0:
            raise ValueError(f"[Ledger] Invalid trade size: {trade_size}")
        if "entry_price" in kwargs:
            entry_price_val = kwargs["entry_price"]
            if not isinstance(entry_price_val, (int, float)) or entry_price_val <= 0:
                raise ValueError(f"[Ledger] Invalid entry_price: {entry_price_val}")

        trade_id = kwargs.get("trade_id") or str(uuid.uuid4())
        entry_price = kwargs.get("entry_price", 18000 + 250 * (0.5 - random.random()))
        # Apply side-aware slippage per CONFIG and round to 4 decimals
        side = kwargs.get("side") or ("buy" if strategy_name.lower().find("sell") == -1 else "sell")
        slip_rate = _slippage_rate_for_pair(trading_pair)
        raw_entry = float(entry_price)
        if side == "buy":
            entry_price = raw_entry * (1 + slip_rate)
            slippage_amount_entry = entry_price - raw_entry
        else:  # sell
            entry_price = raw_entry * (1 - slip_rate)
            slippage_amount_entry = raw_entry - entry_price
        entry_price = round(entry_price, 4)
        slippage_amount_entry = round(slippage_amount_entry, 4)

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
            "cost_basis": round(entry_price * trade_size, 4),
            "entry_price": entry_price,
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

        validate_trade_schema(trade)

        with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(trade) + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

        print(f"[LOGGED] Trade logged: {json.dumps(trade, indent=2)}")
        self.trades.append(trade)
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
        print(f"[DEBUG] Position {trade_id} written to positions.jsonl")

    def update_trade(self, trade_id, exit_price, reason):
        """
        Update a trade with exit information such as exit_price, reason, and ROI.
        If trade is missing, attempts to reload from file and/or reconstruct from positions.
        Retries up to 3 times on failure and safely resyncs positions.jsonl.
        """
        if not isinstance(exit_price, (int, float)) or exit_price <= 0:
            raise ValueError(f"[Ledger] Invalid exit_price for trade {trade_id}: {exit_price}")

        if not os.path.exists(TRADES_LOG_PATH):
            print(f"[Ledger] Trade file not found: {TRADES_LOG_PATH}")
            return

        if not hasattr(self.position_manager, "positions") or self.position_manager.positions is None:
            print(f"[Ledger] position_manager.positions unavailable for trade {trade_id}")
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                updated = False

                # Ensure we have latest trades in memory
                trades = self.trades or []
                if trade_id not in [t.get("trade_id") for t in trades]:
                    trades = self.reload_trades()

                # If still not present, try reconstruct from positions
                if trade_id not in [t.get("trade_id") for t in trades]:
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
                        print(f"[Ledger] Synced missing trade {trade_id} from position")

                # If still not found, log a clear error and stop
                if trade_id not in [t.get("trade_id") for t in trades]:
                    print((f"[Ledger] Trade ID {trade_id} not found in memory or " "trades.log — aborting update"))
                    return

                # Apply update
                for trade in trades:
                    if trade.get("trade_id") == trade_id and trade.get("exit_price") is None:
                        entry_price = trade["entry_price"]
                        size = trade["size"]
                        try:
                            timestamp = datetime.fromisoformat(trade["timestamp"])
                        except ValueError as e:
                            print(
                                (
                                    f"[Ledger] Invalid timestamp for trade {trade_id}:"
                                    f"{trade['timestamp']} - Error: {e}"
                                )
                            )
                            continue

                        # Apply exit slippage (assume closing long -> sell) prior to ROI/gain calc
                        slip_rate = _slippage_rate_for_pair(trade.get("pair"))
                        exit_adj = round(exit_price * (1 - slip_rate), 4)
                        exit_slippage_amount = round(exit_price - exit_adj, 4)
                        trade.update(
                            {
                                "exit_price": exit_adj,
                                "status": "closed",
                                "reason": reason,
                                "holding_period_days": round(
                                    (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400,
                                    2,
                                ),
                                "realized_gain": round((exit_adj - entry_price) * size, 4),
                                "roi": (
                                    round(
                                        (exit_adj - entry_price) / entry_price,
                                        6,
                                    )
                                    if entry_price
                                    else 0.0
                                ),
                                # Slippage metadata
                                "exit_slippage_rate": round(slip_rate, 6),
                                "exit_slippage_amount": exit_slippage_amount,
                            }
                        )
                        updated = True
                        break

                if updated:
                    # Safely rewrite trades.log
                    temp_path = TRADES_LOG_PATH + ".tmp"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            for t in trades:
                                if t is not None:
                                    f.write(json.dumps(t) + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)
                    os.replace(temp_path, TRADES_LOG_PATH)
                    self.trades = trades
                    print(f"[DEBUG] Successfully updated trade {trade_id} in trades.log")

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
                            print(("[DEBUG] Synchronized positions.jsonl — removed closed " "position " f"{trade_id}"))
                    except (OSError, IOError) as e:
                        print((f"[Ledger] Warning: failed to sync positions.jsonl for {trade_id}: " f"{e}"))
                else:
                    # If trade is already closed, treat this as idempotent
                    existing = next(
                        (t for t in trades if t.get("trade_id") == trade_id),
                        None,
                    )
                    if existing and existing.get("exit_price") is not None and existing.get("status") == "closed":
                        print(f"[Ledger] Idempotent update — trade {trade_id} already closed")
                    else:
                        print((f"[Ledger] No update applied — trade_id {trade_id} " "not in open state"))
                return

            except (OSError, IOError, json.JSONDecodeError) as e:
                print(f"[Ledger] Error updating trade (attempt {attempt + 1}): {e}")
                time.sleep(1)

        print(f"[Ledger] Failed to update trade {trade_id} after {max_retries} attempts")

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
                    if line.strip():
                        try:
                            trade = json.loads(line)
                            if trade.get("trade_id"):
                                self.trades.append(trade)
                        except json.JSONDecodeError as e:
                            print(("[Ledger] Skipping invalid line in trades.log:" f"{line.strip()} - Error: {e}"))
            print(f"[Ledger] Reloaded {len(self.trades)} trades from trades.log")
        else:
            print("[Ledger] No trades.log file found.")
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
            print(f"[Ledger] No entries found for trade_id: {target_trade_id}")
            return False
        latest = max(matching, key=lambda x: x.get("timestamp", "1970-01-01T00:00:00+00:00"))
        required_fields = ["status", "exit_price", "reason", "roi"]
        status_ok = latest.get("status") == "closed"
        fields_ok = all(latest.get(f) is not None for f in required_fields)
        if status_ok and fields_ok:
            print(f"[Ledger] Verified update for trade {target_trade_id}")
            return True
        print(f"[Ledger] Trade found but not fully updated: {latest}")
        return False
