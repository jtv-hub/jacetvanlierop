#!/usr/bin/env python3
"""
metrics_snapshot.py — compute basic trading metrics from paper_trades.jsonl

Outputs:
- A console summary
- Appends/creates logs/snapshots/metrics_snapshots.csv with timestamped metrics

Assumptions:
- Events are JSON Lines in logs/paper_trades.jsonl
- We open only long positions ("BUY") and close with "SELL"
- One position at a time (paper_trade.py enforces this)
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGS = Path("logs")
TRADES_FILE = LOGS / "paper_trades.jsonl"
SNAPSHOT_DIR = LOGS / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = SNAPSHOT_DIR / "metrics_snapshots.csv"


def load_events(path: Path) -> List[dict]:
    """Load JSONL trade events from *path*. Malformed/blank lines are skipped."""
    if not path.exists():
        return []
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return out


def pair_trades(events: List[dict]) -> List[dict]:
    """Pair open/close events into closed-trade records.

    Returns a list of dicts with keys:
    timestamp_open, timestamp_close, strategy, entry, exit, size, reason, pnl, win
    """
    position: Optional[dict] = None
    closed: List[dict] = []

    for ev in events:
        etype = ev.get("type")
        if etype == "open":
            # start new position (assume long buys)
            position = {
                "timestamp_open": ev.get("timestamp"),
                "strategy": ev.get("strategy"),
                "entry": float(ev.get("price", 0.0)),
                "size": float(ev.get("size", 0.0)),
            }
        elif etype == "close" and position is not None:
            exit_price = float(ev.get("price", 0.0))
            entry = float(position.get("entry", 0.0))
            size = float(position.get("size", 0.0))
            pnl = (exit_price - entry) * size  # long only

            closed.append(
                {
                    "timestamp_open": position.get("timestamp_open"),
                    "strategy": position.get("strategy"),
                    "entry": entry,
                    "size": size,
                    "timestamp_close": ev.get("timestamp"),
                    "exit": exit_price,
                    "reason": ev.get("reason", ""),
                    "pnl": pnl,
                    "win": pnl > 0.0,
                }
            )
            position = None
        else:
            # ignore other lines
            pass

    return closed


def compute_metrics(
    closed: List[dict],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute overall and per-strategy metrics from closed trades.

    Returns (overall_metrics, by_strategy_metrics). Each metrics dict contains
    keys: count, wins, win_rate, avg_win, avg_loss, expectancy.
    """

    def summarize(trades: List[dict]) -> Dict[str, float]:
        count = len(trades)
        if count == 0:
            return {
                "count": 0.0,
                "wins": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
            }
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
        win_count = len(wins)
        loss_count = len(losses)

        avg_win = sum(wins) / win_count if win_count else 0.0
        avg_loss = sum(losses) / loss_count if loss_count else 0.0
        win_rate = win_count / count if count else 0.0

        # Expectancy ≈ P(win)*AvgWin + P(loss)*AvgLoss
        p_loss = 1.0 - win_rate
        expectancy = win_rate * avg_win + p_loss * avg_loss

        return {
            "count": float(count),
            "wins": float(win_count),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
        }

    overall = summarize(closed)

    bucket: Dict[str, List[dict]] = defaultdict(list)
    for t in closed:
        bucket[t.get("strategy", "unknown")].append(t)

    by_strat: Dict[str, Dict[str, float]] = {}
    for strat, lst in bucket.items():
        by_strat[strat] = summarize(lst)
    return overall, by_strat


def append_csv(
    ts: datetime,
    overall: Dict[str, float],
    by_strat: Dict[str, Dict[str, float]],
) -> None:
    """Append a metrics row to the CSV (creating it with a header if absent)."""
    header = (
        "timestamp,count,wins,win_rate,avg_win,avg_loss,expectancy,"
        "by_strategy_json\n"
    )
    row = (
        f"{ts:%Y-%m-%d %H:%M:%S},"
        f"{int(overall['count'])},"
        f"{int(overall['wins'])},"
        f"{overall['win_rate']:.4f},"
        f"{overall['avg_win']:.2f},"
        f"{overall['avg_loss']:.2f},"
        f"{overall['expectancy']:.2f},"
        f'"{json.dumps(by_strat)}"\n'
    )
    if not CSV_OUT.exists():
        CSV_OUT.write_text(header, encoding="utf-8")
    with CSV_OUT.open("a", encoding="utf-8") as f:
        f.write(row)


def main() -> int:
    """Entry point: compute metrics, print a summary, and append a CSV row."""
    events = load_events(TRADES_FILE)
    closed = pair_trades(events)
    overall, by_strat = compute_metrics(closed)

    # Console output (keep lines ≤ 100 chars)
    closed_str = (
        f"Closed trades: {int(overall['count'])} | Wins: {int(overall['wins'])} "
        f"| Win rate: {overall['win_rate']*100:.1f}%"
    )
    profit_str = (
        f"Avg win: {overall['avg_win']:.2f} | Avg loss: {overall['avg_loss']:.2f} "
        f"| Expectancy/trade: {overall['expectancy']:.2f}"
    )

    print("\n=== METRICS SNAPSHOT ===")
    print(closed_str)
    print(profit_str)
    print("By strategy:")
    for strat, met in by_strat.items():
        # split into two short prints to avoid long line
        left = (
            f"  - {strat}: n={int(met['count'])}, " f"win%={met['win_rate']*100:.1f}%"
        )
        right = f"    exp/trade={met['expectancy']:.2f}"
        print(left)
        print(right)

    ts = datetime.now()
    append_csv(ts, overall, by_strat)
    print(f"\nSaved -> {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
