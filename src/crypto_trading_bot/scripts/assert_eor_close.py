#!/usr/bin/env python3
"""
Assert the last paper-trade event is an End-Of-Run close ("EOR").

Usage:
    scripts/assert_eor_close.py
Exit codes:
    0  = last event is a close with reason "EOR"
    1  = log missing or empty
    2  = last event exists but is not an EOR close
"""

import json
import sys
from pathlib import Path

LOG_PATH = Path("logs/paper_trades.jsonl")


def load_last_event() -> dict | None:
    """Return the last valid JSON object in the paper trades log, or None."""
    if not LOG_PATH.exists():
        return None

    last: dict | None = None
    with LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupt lines and continue
                continue
    return last


def main() -> int:
    """Check that the final event is a close with reason 'EOR' and exit accordingly."""
    last = load_last_event()
    if not last:
        print("❌ No paper_trades.jsonl found or file is empty")
        return 1

    if last.get("type") == "close" and last.get("reason") == "EOR":
        print("✅ Last event is EOR close")
        return 0

    print(f"❌ Last event is not EOR close: {last}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
