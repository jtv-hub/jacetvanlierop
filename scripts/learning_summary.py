"""
learning_summary.py

Summarize recent learning feedback and shadow test results.

Outputs a compact, terminal-friendly dashboard and optionally saves it to
``logs/learning_summary.out`` when ``--save`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

LEARN_FEEDBACK = os.path.join("logs", "learning_feedback.jsonl")
SHADOW_RESULTS = os.path.join("logs", "shadow_test_results.jsonl")
OUT_PATH = os.path.join("logs", "learning_summary.out")


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


def _first(obj: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in obj:
            return obj.get(k)
    return None


def summarize_learning_feedback(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    inc = dec = same = 0
    deltas: List[float] = []

    strategies: List[str] = []
    for r in rows:
        cur = _first(r, "current_confidence", "confidence")
        sug = _first(r, "suggested_confidence", "suggested_value", "value")
        try:
            cur_f = float(cur) if cur is not None else None
            sug_f = float(sug) if sug is not None else None
        except (TypeError, ValueError):
            cur_f = None
            sug_f = None
        if cur_f is not None and sug_f is not None:
            d = sug_f - cur_f
            deltas.append(d)
            if d > 0:
                inc += 1
            elif d < 0:
                dec += 1
            else:
                same += 1

        strat = _first(r, "strategy", "strategy_name") or "Unknown"
        if isinstance(strat, str):
            strategies.append(strat)

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    top3 = [s for s, _ in Counter(strategies).most_common(3)]

    return {
        "total": total,
        "increase": inc,
        "decrease": dec,
        "unchanged": same,
        "avg_delta": avg_delta,
        "top3_strategies": top3,
    }


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def summarize_shadow_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Normalize target fields
    items: List[Tuple[float | None, int | None, Dict[str, Any]]] = []
    for r in rows:
        wr = _first(r, "win_rate", "success_rate")
        ne = _first(r, "num_exits", "trades_tested", "sample_size")
        wr_f = _to_float(wr)
        try:
            ne_i = int(ne) if ne is not None else None
        except (TypeError, ValueError):
            ne_i = None
        items.append((wr_f, ne_i, r))

    recent5 = items[-5:]
    recent5_fmt = [
        {
            "win_rate": (f"{wr:.2f}" if isinstance(wr, float) else "n/a"),
            "num_exits": (ne if isinstance(ne, int) else "n/a"),
        }
        for wr, ne, _ in recent5
    ]

    last10 = items[-10:]
    best_wr = max((wr for wr, _, _ in last10 if isinstance(wr, float)), default=None)
    gt_70 = sum(1 for wr, _, _ in items if isinstance(wr, float) and wr > 0.70)

    return {"recent5": recent5_fmt, "best_win_rate_10": best_wr, "gt_70_count": gt_70}


def format_dashboard(learn: Dict[str, Any], shadow: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("\n===== 📘 Learning Summary =====\n")
    lines.append(f"Suggestions: {learn['total']}")
    lines.append(f"Confidence changes: +{learn['increase']} / -{learn['decrease']} / ={learn['unchanged']}")
    lines.append(f"Avg delta: {learn['avg_delta']:.3f}")
    if learn.get("top3_strategies"):
        lines.append("Top strategies: " + ", ".join(learn["top3_strategies"]))

    lines.append("\n===== 🧪 Shadow Tests =====\n")
    lines.append("Most recent 5:")
    for row in shadow.get("recent5", []):
        lines.append(f"  • win_rate={row['win_rate']}, num_exits={row['num_exits']}")
    best10 = shadow.get("best_win_rate_10")
    best10_txt = f"{best10:.2f}" if isinstance(best10, float) else "n/a"
    lines.append(f"Best win rate (last 10): {best10_txt}")
    lines.append(f">70% win rate count: {shadow.get('gt_70_count', 0)}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize learning and shadow test results")
    parser.add_argument("--save", action="store_true", help="Write summary to logs/learning_summary.out")
    args = parser.parse_args()

    learn_rows = _read_jsonl(LEARN_FEEDBACK)
    shadow_rows = _read_jsonl(SHADOW_RESULTS)

    learn_s = summarize_learning_feedback(learn_rows)
    shadow_s = summarize_shadow_results(shadow_rows)
    out = format_dashboard(learn_s, shadow_s)

    print(out, end="")
    if args.save:
        os.makedirs("logs", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Saved summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
