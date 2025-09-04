"""Validate required fields for closed trades in logs/trades.log (JSONL).

Usage:
  python scripts/validate_closed_trades.py
  python scripts/validate_closed_trades.py --json

Checks each closed trade for:
  - capital_buffer: present and numeric (int or float)
  - side: one of {"long", "short"}
  - roi: present and numeric (int or float)
  - reason: non-empty string

Skips malformed JSON lines with a warning (does not crash).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

TRADES_PATH = os.path.join("logs", "trades.log")
REQUIRED_FIELDS = ("capital_buffer", "side", "roi", "reason")


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _is_numeric(val: Any) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _validate_trade(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return (is_valid, missing_or_invalid_fields) for a closed trade object."""
    missing: List[str] = []

    # capital_buffer: numeric
    if "capital_buffer" not in obj or not _is_numeric(obj.get("capital_buffer")):
        missing.append("capital_buffer")

    # side: "long" or "short"
    side = obj.get("side")
    if not isinstance(side, str) or side.lower() not in {"long", "short"}:
        missing.append("side")

    # roi: numeric
    if "roi" not in obj or not _is_numeric(obj.get("roi")):
        missing.append("roi")

    # reason: non-empty string
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        missing.append("reason")

    return (len(missing) == 0, missing)


def validate_closed_trades(path: str) -> Dict[str, Any]:
    """Validate closed trades and return a summary dictionary."""
    lines = _read_lines(path)
    if not lines:
        return {
            "total_closed_trades": 0,
            "valid": 0,
            "invalid": 0,
            "missing_fields": {},
        }

    total_closed = 0
    valid = 0
    invalid = 0
    missing_counter: Counter[str] = Counter()

    for idx, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            print(f"⚠️  Malformed JSON (L{idx}); skipping")
            continue
        if obj.get("status") != "closed":
            continue

        total_closed += 1
        is_ok, missing = _validate_trade(obj)
        if is_ok:
            valid += 1
        else:
            invalid += 1
            for f in missing:
                missing_counter[f] += 1

    return {
        "total_closed_trades": total_closed,
        "valid": valid,
        "invalid": invalid,
        "missing_fields": dict(missing_counter),
    }


def _print_human(summary: Dict[str, Any]) -> None:
    total = summary["total_closed_trades"]
    valid = summary["valid"]
    invalid = summary["invalid"]
    missing_fields: Dict[str, int] = summary.get("missing_fields", {})

    if total == 0:
        print("ℹ️  No closed trades found in logs/trades.log.")
        return

    if invalid == 0:
        print("✅ All closed trades have required fields populated.")
        print(f"Total closed trades: {total}")
        return

    print(f"Valid closed trades: {valid}")
    print(f"Invalid closed trades: {invalid}")
    if missing_fields:
        print("\nMissing/invalid fields summary (most common first):")
        for field, count in sorted(missing_fields.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {field}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate required fields on closed trades")
    parser.add_argument("--json", action="store_true", help="Return summary as JSON")
    args = parser.parse_args()

    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to validate.")
        return

    summary = validate_closed_trades(TRADES_PATH)
    if args.json:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        _print_human(summary)


if __name__ == "__main__":
    main()
