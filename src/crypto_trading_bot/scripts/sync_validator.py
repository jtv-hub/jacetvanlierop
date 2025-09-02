"""
Sync Validator Module

Validates consistency between trades.log and positions.jsonl.
Includes duplicate checks, timestamps, and repair capability.
"""

import importlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler

# Update this import path based on your project structure
# from crypto_trading_bot.ledger.trade_ledger import TradeLedgerClass

logger = logging.getLogger("sync_validator")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    rotating_handler = get_rotating_handler("validation_errors.log")
    logger.addHandler(rotating_handler)


class SyncValidator:
    """Validates synchronization between trades and positions logs,
    detects mismatches, and optionally repairs using a ledger."""

    def __init__(
        self,
        trades_file: str = os.getenv("TRADES_FILE", "logs/trades.log"),
        positions_file: str = os.getenv("POSITIONS_FILE", "logs/positions.jsonl"),
        log_file: str = os.getenv("VALIDATION_LOG", "logs/validation_errors.log"),
        ledger: Optional[object] = None,  # Replace with TradeLedgerClass once used
    ):
        self.trades_file = trades_file
        self.positions_file = positions_file
        self.log_file = log_file
        self.ledger = ledger
        self.validation_errors: List[str] = []

    def load_json_lines(self, path: str, source: str) -> List[Dict]:
        """Load JSON lines from the specified file.

        Checks for duplicates and malformed entries."""
        entries = []
        seen_ids = set()

        if not os.path.exists(path):
            msg = f"{source} file not found: {path}"
            self.validation_errors.append(msg)
            return entries

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        tid = obj.get("trade_id")
                        if tid in seen_ids:
                            msg = f"Duplicate trade_id in {source}: {tid}"
                            self.validation_errors.append(msg)
                        seen_ids.add(tid)
                        entries.append(obj)
                    except json.JSONDecodeError as e:
                        msg = f"Malformed JSON in {source}: {line.strip()} - {e}"
                        self.validation_errors.append(msg)
        return entries

    def validate_sync(self) -> bool:
        """Validate synchronization between trades.log and positions.jsonl.

        Logs any inconsistencies found."""
        self.validation_errors = []

        trades = self.load_json_lines(self.trades_file, "trades.log")
        positions = self.load_json_lines(self.positions_file, "positions.jsonl")

        trades_by_id = {t["trade_id"]: t for t in trades if "trade_id" in t}
        positions_by_id = {p["trade_id"]: p for p in positions if "trade_id" in p}
        all_ids = set(trades_by_id) | set(positions_by_id)

        for tid in all_ids:
            trade = trades_by_id.get(tid)
            pos = positions_by_id.get(tid)

            if not trade:
                self.validation_errors.append(f"Missing in trades.log: {tid}")

            # Allow closed trades to be absent from positions.jsonl
            if not pos:
                if (
                    trade
                    and trade.get("status") == "closed"
                    and trade.get("exit_price") is not None
                    and trade.get("reason")
                ):
                    # Closed trades are expected to be removed from positions
                    pass
                else:
                    self.validation_errors.append(f"Missing in positions.jsonl: {tid}")

            if trade and pos:
                fields = ["pair", "size", "entry_price", "strategy", "confidence"]
                for field in fields:
                    if trade.get(field) != pos.get(field):
                        msg = (
                            f"Field mismatch for trade_id {tid}: {field} - "
                            f"trade={trade.get(field)} | position={pos.get(field)}"
                        )
                        self.validation_errors.append(msg)

                t_time = trade.get("timestamp")
                p_time = pos.get("timestamp")
                if t_time and p_time:
                    try:
                        t_dt = datetime.fromisoformat(t_time)
                        p_dt = datetime.fromisoformat(p_time)
                        if abs((t_dt - p_dt).total_seconds()) > 60:
                            msg = f"Timestamp mismatch for trade_id {tid}: " f"trade={t_time}, position={p_time}"
                            self.validation_errors.append(msg)
                    except (ValueError, TypeError):
                        # Handle timestamp format or comparison errors gracefully
                        self.validation_errors.append(f"Invalid timestamp format for trade_id {tid}")

                if trade.get("exit_price") is None and trade.get("status") == "closed":
                    self.validation_errors.append(f"Closed trade missing exit_price: {tid}")
                if trade.get("exit_price") and trade.get("status") != "closed":
                    self.validation_errors.append(f"Open trade has exit_price: {tid}")

        self.log_validation_errors(self.log_file)
        return len(self.validation_errors) == 0

    def log_validation_errors(self, log_file: str = "logs/validation_errors.log"):
        """Append validation results to the log file.

        - On errors: write each error line with UTC timestamp
        - On success: write a single 'Sync validation passed' line
        """
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now(timezone.utc).isoformat()
                if self.validation_errors:
                    for error in self.validation_errors:
                        f.write(f"[{timestamp}] {error}\n")
                else:
                    f.write(f"[{timestamp}] ✅ Sync validation passed\n")
        except IOError as e:
            print(f"Failed to write to {log_file}: {e}")

    def repair_missing_trades(self) -> int:
        """Attempt to repair missing trades.

        Recreates entries in the log using position data and the ledger."""
        if not self.ledger:
            self.validation_errors.append("Repair skipped: TradeLedgerClass not provided.")
            self.log_validation_errors(self.log_file)
            return 0

        trades = self.load_json_lines(self.trades_file, "trades.log")
        positions = self.load_json_lines(self.positions_file, "positions.jsonl")
        trades_by_id = {t["trade_id"]: t for t in trades if "trade_id" in t}

        # Ensure ledger writes to the same trades file as validator
        try:
            ledger_module = importlib.import_module(self.ledger.__class__.__module__)
            if getattr(ledger_module, "TRADES_LOG_PATH", None) != self.trades_file:
                setattr(ledger_module, "TRADES_LOG_PATH", self.trades_file)
        except AttributeError as e:
            self.validation_errors.append(f"Warning: could not align ledger TRADES_LOG_PATH: {e}")

        repairs = 0

        for pos in positions:
            tid = pos.get("trade_id")
            if not tid:
                # Skip positions without trade_id
                continue
            if tid not in trades_by_id:
                try:
                    self.ledger.log_trade(
                        trading_pair=pos.get("pair"),
                        trade_size=pos.get("size"),
                        strategy_name=pos.get("strategy", "unknown"),
                        trade_id=tid,
                        entry_price=pos.get("entry_price"),
                        confidence=float(pos.get("confidence", 0.0) or 0.0),
                    )
                    self.validation_errors.append(f"Repaired missing trade: {tid}")
                    repairs += 1
                except (AttributeError, TypeError, ValueError) as e:
                    self.validation_errors.append(f"Repair failed for trade_id {tid}: {e}")
        self.log_validation_errors(self.log_file)
        return repairs


if __name__ == "__main__":
    validator = SyncValidator()
    if not validator.validate_sync():
        print("Validation failed:")
        for validation_error in validator.validation_errors:
            print(f" - {validation_error}")
    else:
        print("✅ Sync validation passed.")
