"""Validate required fields on closed trades in logs/trades.log.

Usage:
    python scripts/validate_trade_fields.py
    python scripts/validate_trade_fields.py --json

Checks that each closed trade has non-empty values for: capital_buffer, side, roi, reason.
Prints a summary and optionally emits JSON of failures.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

TRADES_PATH = os.path.join("logs", "trades.log")
REQUIRED_FIELDS = ["capital_buffer", "side", "roi", "reason"]


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def validate_trades(path: str) -> Dict[str, Any]:
    """Return dict with failures list and pass/fail counts for closed trades."""
    lines = _read_lines(path)
    if not lines:
        return {"closed_total": 0, "passed": 0, "failed": 0, "failures": []}

    failures: List[Dict[str, Any]] = []
    closed_total = 0
    for ln in lines:
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            # skip malformed lines gracefully
            continue
        if obj.get("status") != "closed":
            continue
        closed_total += 1
        missing = [f for f in REQUIRED_FIELDS if _is_empty(obj.get(f))]
        if missing:
            failures.append(
                {
                    "trade_id": obj.get("trade_id"),
                    "missing_fields": missing,
                }
            )

    failed = len(failures)
    passed = max(closed_total - failed, 0)
    return {"closed_total": closed_total, "passed": passed, "failed": failed, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate required fields on closed trades")
    parser.add_argument("--json", action="store_true", help="Output JSON array of failed trades")
    args = parser.parse_args()

    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to validate.")
        return

    result = validate_trades(TRADES_PATH)
    closed_total = result["closed_total"]

    if closed_total == 0:
        print("ℹ️  No closed trades found in logs/trades.log.")
        return

    if args.json:
        print(json.dumps(result["failures"], ensure_ascii=False))
        return

    # Human-readable output
    for item in result["failures"]:
        tid = item.get("trade_id") or "<unknown>"
        missing = ", ".join(item.get("missing_fields") or [])
        print(f"❌ Trade {tid} is missing: [{missing}]")

    if result["failed"] == 0:
        print("✅ All closed trades have required fields populated.")
    else:
        print(f"✅ {result['passed']} trades passed field validation")
        print(f"❌ {result['failed']} trades failed field validation")


if __name__ == "__main__":
    main()
