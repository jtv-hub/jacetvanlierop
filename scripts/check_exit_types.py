"""Check occurrence of required exit types in logs/trades.log.

Usage:
    python scripts/check_exit_types.py
    python scripts/check_exit_types.py --json

Only standard library modules are used: os, json, argparse, collections.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

TRADES_PATH = os.path.join("logs", "trades.log")
REQUIRED = {"sl_triggered", "tp_hit", "indicator_exit", "trailing_exit", "max_hold_expired"}
_DISPLAY_LABELS = {
    "sl_triggered": "STOP_LOSS",
    "tp_hit": "TAKE_PROFIT",
    "indicator_exit": "RSI_EXIT",
    "trailing_exit": "TRAILING_STOP",
    "max_hold_expired": "MAX_HOLD",
}
_LEGACY_MAP = {
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


def _normalize_reason(obj: dict) -> str | None:
    """Return canonical exit reason for a trade record."""

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
        if lowered in REQUIRED:
            return lowered
        legacy = text.upper()
        mapped = _LEGACY_MAP.get(legacy)
        if mapped:
            return mapped
    return None


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def compute_exit_counts(path: str) -> tuple[Counter, list[str]]:
    """Return (counts, warnings) for required exit types.

    Counts unknown or unmapped reasons as nothing; malformed JSON lines
    are reported in warnings.
    """
    warnings: list[str] = []
    counts: Counter = Counter()
    lines = _read_lines(path)
    for i, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            warnings.append(f"Malformed JSON (L{i}): {e}")
            continue
        if (obj.get("status") or "").lower() != "closed":
            continue
        norm = _normalize_reason(obj)
        if norm is None:
            continue
        counts[norm] += 1
    # Ensure all REQUIRED keys exist (with zero default)
    for key in REQUIRED:
        counts.setdefault(key, 0)
    return counts, warnings


def main() -> None:
    """CLI entry point: parse file, tally exits, and print results or JSON."""
    parser = argparse.ArgumentParser(description="Check required exit types usage in trades.log")
    parser.add_argument("--json", action="store_true", help="Output counts as JSON")
    args = parser.parse_args()

    if not os.path.exists(TRADES_PATH):
        print("⚠️  logs/trades.log not found. Nothing to check.")
        return

    counts, warnings = compute_exit_counts(TRADES_PATH)

    total_used = sum(counts.values())
    if total_used == 0:
        print("⚠️  No valid closed trades with required exit reasons were found.")
        return

    if args.json:
        print(json.dumps({k: counts.get(k, 0) for k in sorted(REQUIRED)}, separators=(",", ":")))
        return

    for key in sorted(REQUIRED):
        c = counts.get(key, 0)
        icon = "✅" if c > 0 else "❌"
        label = _DISPLAY_LABELS.get(key, key)
        print(f"{icon} {label}: {c}")

    for w in warnings:
        print(f"⚠️  {w}")


if __name__ == "__main__":
    main()
