# scripts/backfill_trade_ledger.py

"""
Backfill Script for Trade Ledger
Ensures all trades in trade_ledger.json contain the new derived fields.
Creates a backup before overwriting.
"""

import json
import os
import shutil
from datetime import datetime

LEDGER_PATH = "ledger/trade_ledger.json"
BACKUP_PATH = "ledger/trade_ledger_backup.json"


def compute_backfill_fields(trade: dict) -> dict:
    """
    Compute and backfill missing fields in a trade entry.
    """
    trade_size = trade.get("trade_size", 0.0)
    pnl_pct = trade.get("pnl_pct", 0.0)

    # PNL & ROI
    pnl = round(trade_size * pnl_pct, 4)
    roi = round(pnl_pct * 100, 4)
    status = "win" if pnl_pct > 0 else "loss"

    # Trade duration
    trade_duration = 0.0
    ts_start = trade.get("ts_start", "")
    ts_end = trade.get("ts_end", "")
    if ts_start and ts_end:
        try:
            start = datetime.fromisoformat(ts_start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(ts_end.replace("Z", "+00:00"))
            trade_duration = (end - start).total_seconds() / 60.0
        except (ValueError, TypeError):
            trade_duration = 0.0

    # Backfill missing fields
    trade.setdefault("pnl", pnl)
    trade.setdefault("roi", roi)
    trade.setdefault("status", status)
    trade.setdefault("trade_duration", trade_duration)
    trade.setdefault("vol_entry", 0.0)
    trade.setdefault("vol_exit", 0.0)

    return trade


def backfill_ledger():
    """
    Loads trade_ledger.json, backfills missing fields, and saves updates.
    """
    if not os.path.exists(LEDGER_PATH):
        print("❌ No trade_ledger.json found.")
        return

    # Backup original file
    shutil.copy(LEDGER_PATH, BACKUP_PATH)
    print(f"📦 Backup created at {BACKUP_PATH}")

    with open(LEDGER_PATH, "r", encoding="utf-8") as file:
        try:
            trades = json.load(file)
        except json.JSONDecodeError:
            print("❌ Could not parse trade_ledger.json")
            return

    # Process trades
    upgraded_trades = [compute_backfill_fields(trade) for trade in trades]

    # Save updated ledger
    with open(LEDGER_PATH, "w", encoding="utf-8") as file:
        json.dump(upgraded_trades, file, indent=2)

    print(f"✅ Backfilled {len(upgraded_trades)} trades and updated {LEDGER_PATH}")


if __name__ == "__main__":
    backfill_ledger()
