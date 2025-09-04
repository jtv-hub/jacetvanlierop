"""Generate learning suggestions from the fixed trade log.

Reads `logs/trades_fixed.log` (fallbacks to `logs/trades.log` if the fixed
file is missing), scans for loss patterns or low ROI, and writes suggestion
records to `logs/learning_feedback.jsonl` with fields:

- timestamp
- trade_id
- current_confidence
- suggested_confidence
- reason

Ensures only one suggestion per trade per run, and skips trades that already
have a suggestion in the output file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, Set

TRADES_FIXED = os.path.join("logs", "trades_fixed.log")
TRADES_ORIG = os.path.join("logs", "trades.log")
OUTPUT_PATH = os.path.join("logs", "learning_feedback.jsonl")


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
                continue


def _existing_suggestion_ids() -> Set[str]:
    ids: Set[str] = set()
    for obj in _load_jsonl(OUTPUT_PATH):
        tid = obj.get("trade_id")
        if isinstance(tid, str) and tid:
            ids.add(tid)
    return ids


def _choose_source() -> str:
    return TRADES_FIXED if os.path.exists(TRADES_FIXED) else TRADES_ORIG


def _generate_suggestion(trade: Dict) -> dict | None:
    """Return a suggestion dict or None.

    Very simple rules:
    - If ROI < 0: reduce confidence.
    - If small gain (0 <= ROI < 0.01) and high confidence (>0.8): nudge down.
    """
    try:
        roi = float(trade.get("roi"))
    except (TypeError, ValueError):
        return None

    cur_conf = trade.get("confidence")
    try:
        cur_conf_f = float(cur_conf) if cur_conf is not None else 0.0
    except (TypeError, ValueError):
        cur_conf_f = 0.0

    tid = trade.get("trade_id")
    if not isinstance(tid, str) or not tid:
        return None

    if roi < 0:
        suggested = max(0.1, round(min(cur_conf_f, 0.6) - 0.2, 2))
        reason = f"Loss trade (roi={roi:.4f}); reduce confidence"
    elif 0 <= roi < 0.01 and cur_conf_f > 0.8:
        suggested = round(cur_conf_f - 0.1, 2)
        reason = f"Low ROI with high confidence (roi={roi:.4f})"
    else:
        return None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trade_id": tid,
        "current_confidence": round(cur_conf_f, 4),
        "suggested_confidence": suggested,
        "reason": reason,
    }


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    src = _choose_source()
    existing = _existing_suggestion_ids()

    wrote = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        seen_this_run: Set[str] = set()
        for trade in _load_jsonl(src):
            sug = _generate_suggestion(trade)
            if not sug:
                continue
            tid = sug["trade_id"]
            if tid in seen_this_run or tid in existing:
                continue
            out.write(json.dumps(sug, separators=(",", ":")) + "\n")
            seen_this_run.add(tid)
            wrote += 1

    print(f"Wrote {wrote} new suggestions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
