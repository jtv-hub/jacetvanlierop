"""
shadow_tester.py

Lightweight shadow tester that appends results to logs/shadow_test_results.jsonl.

Heuristic:
- Uses closed trades from logs/trades.log to compute a quick win_rate over the
  last N exits and writes a single JSONL result with summary fields.

This provides data for the Streamlit dashboard when running paper cycles.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any, Dict, List


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def run_shadow_test(
    trades_path: str = "logs/trades.log", out_path: str = "logs/shadow_test_results.jsonl", window: int = 20
) -> Dict[str, Any]:
    os.makedirs("logs", exist_ok=True)
    trades = _read_jsonl(trades_path)
    closed = [t for t in trades if (t.get("status") or "").lower() == "closed"]
    recent = closed[-window:]
    rois = [_to_float(t.get("roi")) for t in recent]
    rois = [r for r in rois if r is not None]
    wins = sum(1 for r in rois if r > 0)
    total = len(rois)
    win_rate = (wins / total) if total else 0.0
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "strategy": recent[-1].get("strategy") if recent else "Unknown",
        "win_rate": round(win_rate, 4),
        "num_exits": total,
        "duration": f"last {min(window, len(closed))} closed",
        "status": "pass" if win_rate >= 0.5 else "fail",
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, separators=(",", ":")) + "\n")
    return result


def main() -> None:
    r = run_shadow_test()
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
