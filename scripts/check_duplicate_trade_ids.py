"""Detect duplicate trade_id values in logs/trades.log.

Usage:
  python scripts/check_duplicate_trade_ids.py
  python scripts/check_duplicate_trade_ids.py --json

Only standard libraries are used (os, json, argparse, collections).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

TRADES_PATH = os.path.join("logs", "trades.log")


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def find_duplicate_trade_ids(path: str) -> tuple[dict[str, int], int, int]:
    """Return (duplicates_map, malformed_count, missing_id_count)."""
    lines = _read_lines(path)
    ids: list[str] = []
    malformed = 0
    missing = 0
    for ln in lines:
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            malformed += 1
            continue
        tid = obj.get("trade_id")
        if not tid:
            missing += 1
            continue
        ids.append(str(tid))

    ctr = Counter(ids)
    duplicates = {k: v for k, v in ctr.items() if v > 1}
    return duplicates, malformed, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for duplicate trade_id values in trades.log")
    parser.add_argument("--json", action="store_true", help="Output duplicates as JSON")
    args = parser.parse_args()

    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to check.")
        return

    dups, malformed, missing = find_duplicate_trade_ids(TRADES_PATH)

    if args.json:
        print(json.dumps(dups, separators=(",", ":")))
        return

    if not dups:
        print("✅ No duplicate trade_id values found.")
    else:
        print("❌ Duplicate trade_id values detected:")
        for tid, count in sorted(dups.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {tid}: {count}")

    if malformed or missing:
        print(f"⚠️  Skipped {malformed} malformed lines, {missing} lines without trade_id.")


if __name__ == "__main__":
    main()
