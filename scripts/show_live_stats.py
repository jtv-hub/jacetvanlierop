"""Show live summary statistics from logs/trades.log (compact JSONL).

Usage:
    python scripts/show_live_stats.py
    python scripts/show_live_stats.py --json

The script reads UTF-8 JSONL, skips malformed lines, and only includes
trades whose status == "closed" with numeric ROI.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

TRADES_PATH = os.path.join("logs", "trades.log")


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _load_closed_trades(path: str) -> List[Dict[str, Any]]:
    """Load closed trades with numeric ROI; skip malformed or incomplete."""
    rows: List[Dict[str, Any]] = []
    for ln in _read_lines(path):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if obj.get("status") != "closed":
            continue
        roi = obj.get("roi")
        try:
            obj["roi"] = float(roi)
        except (TypeError, ValueError):
            continue
        rows.append(obj)
    return rows


def _fmt_pct(x: float, places: int = 2, sign: bool = True) -> str:
    s = f"{x*100:.{places}f}%"
    if sign and not s.startswith("-"):
        s = "+" + s
    return s


def _compute_summary(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trades)
    wins = sum(1 for t in trades if t["roi"] > 0)
    losses = total - wins
    win_rate = (wins / total) if total else 0.0
    cum_roi = sum(t["roi"] for t in trades) if total else 0.0
    avg_roi = (cum_roi / total) if total else 0.0

    # Strategy leaderboards
    wins_by_strategy: Counter[str] = Counter()
    roi_by_strategy: defaultdict[str, float] = defaultdict(float)
    for t in trades:
        strat = str(t.get("strategy") or "Unknown")
        wins_by_strategy[strat] += 1 if t["roi"] > 0 else 0
        roi_by_strategy[strat] += float(t["roi"])

    # Build combined leaderboard entries (wins, roi)
    combined: List[Dict[str, Any]] = []
    for strat in set(wins_by_strategy) | set(roi_by_strategy):
        combined.append(
            {
                "strategy": strat,
                "wins": int(wins_by_strategy.get(strat, 0)),
                "roi": float(roi_by_strategy.get(strat, 0.0)),
            }
        )
    top = sorted(combined, key=lambda x: (x["wins"], x["roi"]), reverse=True)[:5]

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "cumulative_roi": round(cum_roi, 6),
        "average_roi": round(avg_roi, 6),
        "top_strategies": top,
    }


def _print_human(summary: Dict[str, Any]) -> None:
    print("\n📊 Trade Summary\n")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']} | Losses: {summary['losses']}")
    wr = _fmt_pct(summary["win_rate"], places=1, sign=False)
    print(f"Win Rate: {wr}")
    print(f"Cumulative ROI: {_fmt_pct(summary['cumulative_roi'])}")
    print(f"Average ROI/Trade: {_fmt_pct(summary['average_roi'])}")

    if summary.get("top_strategies"):
        print("\n🏆 Top Strategies")
        for item in summary["top_strategies"]:
            s_name = item["strategy"]
            s_wins = item["wins"]
            s_roi = _fmt_pct(item["roi"]) if isinstance(item["roi"], (int, float)) else str(item["roi"])
            print(f"\t•\t{s_name}: {s_wins} wins, {s_roi}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live stats from trades.log")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON summary")
    args = parser.parse_args()

    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to summarize.")
        return

    trades = _load_closed_trades(TRADES_PATH)
    if not trades:
        print("ℹ️  No closed trades with numeric ROI found.")
        return

    summary = _compute_summary(trades)
    if args.json:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        _print_human(summary)


if __name__ == "__main__":
    main()
