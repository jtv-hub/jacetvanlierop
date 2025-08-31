"""
Paper P&L report (FIFO matching of opens→closes, long-only).

Reads `logs/paper_trades.jsonl` and reports realized P&L overall and
by strategy.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

LEDGER = Path("logs/paper_trades.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class Position:
    """A single open lot awaiting matching against closes."""

    size: float
    entry_price: float
    timestamp: str


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt:
                continue
            try:
                rows.append(json.loads(txt))
            except json.JSONDecodeError:
                continue
    return rows


def compute_pnl(rows: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """
    Compute realized P&L using FIFO lot matching (long-only).

    Returns:
        total_pnl: Overall realized P&L.
        by_strategy: Mapping of strategy name to realized P&L.
    """
    queues: Dict[Tuple[str, str], Deque[Position]] = defaultdict(deque)
    realized = 0.0
    by_strategy: Dict[str, float] = defaultdict(float)

    for r in rows:
        typ = r.get("type")
        pair = r.get("pair")
        strat = r.get("strategy")
        key = (pair, strat)

        if typ == "open":
            queues[key].append(
                Position(
                    size=float(r["size"]),
                    entry_price=float(r["price"]),
                    timestamp=r.get("timestamp", ""),
                )
            )

        elif typ == "close":
            remaining = float(r["size"])
            exit_price = float(r["price"])
            while remaining > 1e-12 and queues[key]:
                pos = queues[key][0]
                take = min(remaining, pos.size)
                pnl = (exit_price - pos.entry_price) * take  # long-only
                realized += pnl
                by_strategy[strat] += pnl
                pos.size -= take
                remaining -= take
                if pos.size <= 1e-12:
                    queues[key].popleft()

    return realized, dict(by_strategy)


def main() -> None:
    """Print a simple P&L report to stdout."""
    rows = _read_jsonl(LEDGER)
    total, by_strat = compute_pnl(rows)

    print("\n=== PAPER P&L REPORT ===")
    print(f"Total realized P&L: {total:.2f}")
    if by_strat:
        print("By strategy:")
        for name, value in by_strat.items():
            print(f"  - {name}: {value:.2f}")
    else:
        print("(no closed positions yet)")
    print("")


if __name__ == "__main__":
    main()
