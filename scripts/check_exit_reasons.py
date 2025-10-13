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
LEGACY_MAP = {
    "STOP_LOSS": "sl_triggered",
    "STOPLOSS": "sl_triggered",
    "TRAILING_STOP": "trailing_exit",
    "TRAIL_STOP": "trailing_exit",
    "TAKE_PROFIT": "tp_hit",
    "TAKEPROFIT": "tp_hit",
    "RSI_EXIT": "indicator_exit",
    "RSI": "indicator_exit",
    "MAX_HOLD": "max_hold_expired",
    "MAX HOLD": "max_hold_expired",
}
DISPLAY_LABELS = {
    "sl_triggered": "Stop Loss",
    "tp_hit": "Take Profit",
    "indicator_exit": "Indicator Exit",
    "trailing_exit": "Trailing Stop",
    "max_hold_expired": "Max Hold",
    "unknown": "Unknown",
}


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _canonical_reason(obj: dict) -> str:
    candidates = [
        obj.get("exit_reason"),
        obj.get("reason"),
        obj.get("reason_display"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        lowered = text.lower().replace(" ", "_")
        if lowered in DISPLAY_LABELS:
            return lowered
        mapped = LEGACY_MAP.get(text.upper())
        if mapped:
            return mapped
    return "unknown"


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
        if (obj.get("status") or "").lower() != "closed":
            continue
        counts[_canonical_reason(obj)] += 1
    return counts


def _print_human(counts: Dict[str, int]) -> None:
    print("\n📤 Exit Reason Summary\n")
    for key, value in counts.items():
        label = DISPLAY_LABELS.get(key, key)
        print(f"• {label}: {value}")


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
