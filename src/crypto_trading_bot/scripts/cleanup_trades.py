"""
cleanup_trades.py

This script scans the logs/trades.log file for invalid or malformed trades —
specifically trades with non-numeric or out-of-bound confidence values —
and removes them. A backup of the original file is saved before cleanup.
"""

import json
import os
from datetime import datetime

TRADE_LOG_PATH = "logs/trades.log"
BACKUP_DIR = "logs/"


def is_valid_confidence(confidence):
    """Check if the confidence value is a valid float between 0.0 and 1.0."""
    try:
        value = float(confidence)
        return 0.0 <= value <= 1.0
    except (ValueError, TypeError):
        return False


def load_trades(path):
    """Load all trades from the log file and return them as a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line.strip()) for line in file if line.strip()]


def save_backup(trades):
    """Save a backup of the original trades log with a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"trades_backup_{timestamp}.log")
    with open(backup_path, "w", encoding="utf-8") as file:
        for trade in trades:
            json.dump(trade, file)
            file.write("\n")
    print(f"🔄 Backup saved to: {backup_path}")


def save_cleaned_trades(trades):
    """Write the cleaned list of valid trades back to the trade log file."""
    with open(TRADE_LOG_PATH, "w", encoding="utf-8") as file:
        for trade in trades:
            json.dump(trade, file)
            file.write("\n")
    print(f"✅ Cleaned trades written to {TRADE_LOG_PATH}")


def cleanup_trade_log():
    """Perform cleanup by filtering out invalid trades and saving the cleaned data."""
    trades = load_trades(TRADE_LOG_PATH)
    save_backup(trades)

    cleaned = [t for t in trades if is_valid_confidence(t.get("confidence"))]
    removed_count = len(trades) - len(cleaned)

    save_cleaned_trades(cleaned)

    print(f"🧹 Removed {removed_count} invalid trades.")


if __name__ == "__main__":
    cleanup_trade_log()
