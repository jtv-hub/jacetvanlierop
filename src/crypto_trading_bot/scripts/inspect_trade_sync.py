"""
inspect_trade_sync.py

Utility to inspect and compare the contents of trades.log and positions.jsonl
to ensure synchronization of trade IDs and entry prices.
"""

import json
import os
from datetime import datetime as _dt

TRADES_LOG_PATH = "logs/trades.log"
POSITIONS_PATH = "logs/positions.jsonl"


def load_json_lines(filepath):
    """Loads a JSON lines file into a list of dictionaries."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def inspect_sync():
    """Inspects sync between trades.log and positions.jsonl."""
    trades = load_json_lines(TRADES_LOG_PATH)
    positions = load_json_lines(POSITIONS_PATH)

    trade_map = {t["trade_id"]: t for t in trades}
    position_map = {p["trade_id"]: p for p in positions}

    errors = 0
    print(f"\n📄 Total trades: {len(trade_map)}, Total positions: {len(position_map)}")

    # Check for missing positions
    for tid in trade_map:
        if tid not in position_map:
            print(f"❌ Position missing for trade_id: {tid}")
            errors += 1

    # Check for missing trades
    for tid in position_map:
        if tid not in trade_map:
            print(f"❌ Trade missing for position trade_id: {tid}")
            errors += 1

    # Compare entry prices
    for tid in trade_map:
        if tid in position_map:
            t_price = trade_map[tid].get("entry_price")
            p_price = position_map[tid].get("entry_price")
            if round(float(t_price), 4) != round(float(p_price), 4):
                print(
                    f"⚠️ Price mismatch for {tid}: trades.log={t_price}, positions.jsonl={p_price}"
                )
                errors += 1

    if errors == 0:
        ts = _dt.now().isoformat()
        print(f"\n✅ All entries are synchronized as of {ts}.\n")
    else:
        print(f"\n❌ Found {errors} sync issues.\n")


if __name__ == "__main__":
    inspect_sync()
