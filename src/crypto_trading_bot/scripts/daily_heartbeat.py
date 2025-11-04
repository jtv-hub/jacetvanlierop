"""
Daily Heartbeat Script

Runs scheduled daily maintenance tasks for the crypto trading bot,
including reinvestment rate refresh
and other periodic health checks or cleanup routines.
"""

import json
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from crypto_trading_bot.bot.state.portfolio_state import (
    get_reinvestment_rate,
)
from crypto_trading_bot.bot.trading_logic import position_manager
from crypto_trading_bot.ledger.trade_ledger import TradeLedger

HEARTBEAT_LOG = "logs/daily_heartbeat.log"
ANOMALIES_LOG = "logs/anomalies.log"


def _log_setup():
    import logging

    logger = logging.getLogger("daily_heartbeat")
    if logger.hasHandlers():
        return logger
    os.makedirs("logs", exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        filename=HEARTBEAT_LOG,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    # keep console output for interactive runs
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(stream)
    return logger


def _append_anomaly(message: str, context: dict | None = None):
    try:
        os.makedirs("logs", exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "ERROR",
            "module": "daily_heartbeat",
            "message": message,
        }
        if context:
            payload["context"] = context
        with open(ANOMALIES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def _is_appendable(path: str) -> tuple[bool, str | None]:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("")
            f.flush()
            os.fsync(f.fileno())
        return True, None
    except OSError as exc:
        return False, str(exc)


def run_daily_tasks():
    """Run all daily scheduled maintenance tasks for the trading bot."""
    logger = _log_setup()
    logger.info("Running daily heartbeat tasks…")

    try:
        # Refresh reinvestment rate (automatically refreshes if outdated)
        rate = get_reinvestment_rate()
        logger.info("Reinvestment rate refreshed: %.0f%%", rate * 100)

        # File health checks
        targets = [
            "logs/trades.log",
            "logs/positions.jsonl",
            "logs/exit_check.log",
            "logs/shadow_test_results.jsonl",
            "logs/learning_feedback.jsonl",
        ]
        problems = []
        for path in targets:
            ok, err = _is_appendable(path)
            if not ok:
                problems.append({"path": path, "error": err})
        if problems:
            _append_anomaly("File health check failures", {"problems": problems})
            logger.error("File health check failures: %s", problems)
        else:
            logger.info("All target files appendable")

        # Ledger consistency check
        position_manager.load_positions_from_file()
        ledger = TradeLedger(position_manager)
        ledger.reload_trades()
        required_fields = {"trade_id", "confidence"}
        open_count = 0
        closed_count = 0
        missing_fields = []
        try:
            for rec in ledger.trades:
                status = (rec.get("status") or "").lower()
                if status == "closed":
                    closed_count += 1
                else:
                    open_count += 1
                missing = [k for k in required_fields if rec.get(k) is None]
                if missing:
                    missing_fields.append({"trade_id": rec.get("trade_id"), "missing": missing})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _append_anomaly("Ledger iteration error", {"error": str(exc)})
            logger.exception("Ledger iteration error")
        logger.info("Ledger summary: open=%d closed=%d", open_count, closed_count)
        if missing_fields:
            _append_anomaly("Ledger missing fields detected", {"records": missing_fields[:25]})
            logger.error("Ledger missing fields in %d records (showing up to 25)", len(missing_fields))

        logger.info("Daily tasks completed successfully.")

    except Exception:  # pylint: disable=broad-except
        logger.exception("Error during daily heartbeat")
        _append_anomaly("Unhandled exception in daily heartbeat", {})


if __name__ == "__main__":
    run_daily_tasks()
