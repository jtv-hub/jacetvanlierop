"""
backfill_side_field.py

Normalize the `side` field in logs/trades.log from legacy values
('long'/'short') to audit-compatible values ('buy'/'sell').

Process:
  1) Stream-read logs/trades.log line by line (JSONL)
  2) Convert side values when present
  3) Write to logs/trades_cleaned.log
  4) Backup original as logs/trades_original.log
  5) Replace logs/trades.log with the cleaned file

Usage:
  python scripts/backfill_side_field.py
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Tuple

IN_PATH = os.path.join("logs", "trades.log")
OUT_PATH = os.path.join("logs", "trades_cleaned.log")
BAK_PATH = os.path.join("logs", "trades_original.log")


def _normalize_side(val: object) -> Tuple[object, bool]:
    """Return (new_value, changed) for a side value."""
    if isinstance(val, str):
        s = val.strip().lower()
        if s == "long":
            return "buy", True
        if s == "short":
            return "sell", True
    return val, False


def backfill() -> int:
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(IN_PATH):
        print(f"ℹ️  {IN_PATH} not found; nothing to backfill.")
        return 0

    updated = 0
    total = 0
    with open(IN_PATH, "r", encoding="utf-8") as src, open(OUT_PATH, "w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            if not line.strip():
                dst.write(line)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # preserve malformed or non-JSON lines as-is
                dst.write(line)
                continue

            side_val = obj.get("side")
            new_val, changed = _normalize_side(side_val)
            if changed:
                obj["side"] = new_val
                updated += 1
            # write compact JSONL
            dst.write(json.dumps(obj, separators=(",", ":")) + "\n")

    # Backup and replace
    try:
        # If a previous backup exists, overwrite it
        shutil.copyfile(IN_PATH, BAK_PATH)
    except OSError as e:
        print(f"⚠️  Failed to create backup {BAK_PATH}: {e}")
    try:
        os.replace(OUT_PATH, IN_PATH)
    except OSError as e:
        print(f"⚠️  Failed to replace {IN_PATH} with cleaned log: {e}")

    print(f"✅ Backfill complete: {updated} lines updated, backup saved to {BAK_PATH}.")
    return updated


if __name__ == "__main__":
    backfill()
