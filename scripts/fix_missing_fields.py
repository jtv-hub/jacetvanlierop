"""Fix missing fields in trades.log.

Scans `logs/trades.log` for trades where status == "closed" but `side` is
missing or null. Attempts to backfill from `logs/positions.jsonl` by
matching `trade_id`. If no match is found, defaults `side` to "buy".

Writes the corrected stream to `logs/trades_fixed.log` and prints a concise
summary of before/after counts.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable

TRADES_PATH = os.path.join("logs", "trades.log")
POSITIONS_PATH = os.path.join("logs", "positions.jsonl")
OUTPUT_PATH = os.path.join("logs", "trades_fixed.log")


def _load_jsonl(path: str) -> Iterable[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines silently; the user can inspect manually if needed
                continue


def _positions_by_id() -> Dict[str, dict]:
    idx: Dict[str, dict] = {}
    for obj in _load_jsonl(POSITIONS_PATH):
        tid = obj.get("trade_id")
        if isinstance(tid, str) and tid:
            idx[tid] = obj
    return idx


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    positions_index = _positions_by_id()

    total = 0
    missing_before = 0
    fixed = 0

    timestamp = datetime.now(timezone.utc).isoformat()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for obj in _load_jsonl(TRADES_PATH):
            total += 1
            status = (obj.get("status") or "").lower()
            side = obj.get("side")
            needs_fix = status == "closed" and (side is None or side == "")
            if needs_fix:
                missing_before += 1
                tid = obj.get("trade_id")
                pos = positions_index.get(tid)
                if pos and pos.get("side"):
                    obj["side"] = pos.get("side")
                else:
                    obj["side"] = "buy"
                fixed += 1

            # Emit compact JSONL
            out.write(json.dumps(obj, separators=(",", ":")) + "\n")

    print(f"Fixed {fixed} missing side fields (before={missing_before}, total={total}) at {timestamp}")


if __name__ == "__main__":
    main()
