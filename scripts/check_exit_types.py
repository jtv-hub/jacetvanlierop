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
REQUIRED = {"STOP_LOSS", "TAKE_PROFIT", "RSI_EXIT", "TRAILING_STOP", "MAX_HOLD"}


def _normalize_reason(raw: str) -> str | None:
    """Map a free-form reason string to one of the REQUIRED categories.

    Returns the normalized reason or None if it cannot be mapped.
    """
    r = (raw or "").upper()
    if not r:
        return None
    if "TRAIL" in r:
        return "TRAILING_STOP"
    if "STOP" in r:
        return "STOP_LOSS"
    if "TAKE" in r or "PROFIT" in r:
        return "TAKE_PROFIT"
    if "RSI" in r:
        return "RSI_EXIT"
    if "MAX_HOLD" in r or "HOLD" in r:
        return "MAX_HOLD"
    if r in REQUIRED:
        return r
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
        if obj.get("status") != "closed":
            continue
        norm = _normalize_reason(str(obj.get("reason", "")))
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
        print(f"{icon} {key}: {c}")

    for w in warnings:
        print(f"⚠️  {w}")


if __name__ == "__main__":
    main()
