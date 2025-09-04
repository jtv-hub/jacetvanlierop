"""Analyze exit reasons in logs/trades.log and print a summary.

Usage:
  python scripts/check_exit_reasons.py
  python scripts/check_exit_reasons.py --json

Only built-in libraries are used.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict

TRADES_PATH = os.path.join("logs", "trades.log")


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def count_exit_reasons(path: str) -> Counter:
    """Return a Counter of exit reasons for closed trades."""
    counts: Counter = Counter()
    for ln in _read_lines(path):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            # Skip malformed
            continue
        if obj.get("status") != "closed":
            continue
        r = obj.get("reason")
        if isinstance(r, str) and r.strip():
            counts[r.strip().upper()] += 1
    return counts


def _print_human(counts: Dict[str, int]) -> None:
    print("\n📤 Exit Reason Summary\n")
    for key, value in counts.items():
        print(f"• {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exit reasons in trades.log")
    parser.add_argument("--json", action="store_true", help="Emit JSON dictionary of counts")
    args = parser.parse_args()

    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to analyze.")
        return

    counts = count_exit_reasons(TRADES_PATH)
    if not counts:
        print("No closed trades with exit reasons found.")
        return

    if args.json:
        print(json.dumps(dict(counts), separators=(",", ":")))
    else:
        _print_human(dict(counts))


if __name__ == "__main__":
    main()
