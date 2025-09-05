"""
Shadow Confidence Test

Simulate alternative confidence thresholds on historical closed trades from
logs/trades.log and append summary metrics to logs/shadow_test_results.jsonl.

Usage:
  python -m src.crypto_trading_bot.scripts.shadow_confidence_test
  or
  python -m scripts.shadow_confidence_test  (if PYTHONPATH includes src)

Outputs JSONL entries like:
  {
    "timestamp": "2025-09-05T17:06:37.998696+00:00",
    "type": "confidence_threshold_eval",
    "threshold": 0.6,
    "trades": 52,
    "win_rate": 0.71,
    "cumulative_roi": 0.193,
    "avg_roi": 0.012,
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterable, List

TRADES_LOG = os.path.join("logs", "trades.log")
OUTPUT_LOG = os.path.join("logs", "shadow_test_results.jsonl")


def _read_jsonl(path: str) -> Iterable[dict]:
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


def _closed_with_confidence(rows: Iterable[dict]) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        if (r.get("status") or "").lower() != "closed":
            continue
        try:
            _ = float(r.get("roi"))
            _ = float(r.get("confidence"))
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def evaluate_threshold(rows: List[dict], threshold: float) -> dict:
    selected = [r for r in rows if (float(r.get("confidence", 0.0)) >= threshold)]
    n = len(selected)
    if n == 0:
        return {
            "threshold": threshold,
            "trades": 0,
            "win_rate": 0.0,
            "cumulative_roi": 0.0,
            "avg_roi": 0.0,
        }
    rois = [float(r.get("roi", 0.0)) for r in selected]
    wins = sum(1 for x in rois if x > 0)
    win_rate = wins / n
    avg_roi = sum(rois) / n
    cumulative_roi = 1.0
    for x in rois:
        cumulative_roi *= 1.0 + x
    cumulative_roi -= 1.0
    return {
        "threshold": threshold,
        "trades": n,
        "win_rate": round(win_rate, 6),
        "cumulative_roi": round(cumulative_roi, 6),
        "avg_roi": round(avg_roi, 6),
    }


def run_shadow_confidence_test(output_path: str = OUTPUT_LOG, thresholds: List[float] | None = None) -> int:
    if thresholds is None:
        thresholds = [round(x, 1) for x in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    rows = _closed_with_confidence(_read_jsonl(TRADES_LOG))
    ts = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wrote = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for th in thresholds:
            res = evaluate_threshold(rows, th)
            rec = {"timestamp": ts, "type": "confidence_threshold_eval", **res}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            wrote += 1
    print(f"[shadow_confidence_test] Wrote {wrote} entries to {output_path}")
    return wrote


def main() -> None:
    run_shadow_confidence_test()


if __name__ == "__main__":
    main()
