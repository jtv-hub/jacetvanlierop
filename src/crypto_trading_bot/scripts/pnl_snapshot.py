#!/usr/bin/env python3
"""
pnl_snapshot.py

Aggregate the current paper ledger and append a one-line snapshot CSV with:
timestamp, total_realized_pnl, total_open_pnl, by_strategy(JSON)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Paths
LEDGER_PATH = Path("logs/paper_trades.jsonl")
SNAPSHOT_DIR = Path("logs/snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_CSV = SNAPSHOT_DIR / "paper_pnl_snapshots.csv"


@dataclass
class Position:
    """Represents a single open long position used in the P&L calc."""

    strategy: str
    entry: float
    size: float


def _read_ledger(path: Path) -> List[dict]:
    """Read jsonl trade events from the paper ledger."""
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return rows


def _compute_pnl(entries: List[dict]) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute realized and open P&L using a very simple FIFO long-only model.

    Returns:
        realized_total: float
        open_pnl_total: float
        by_strategy_realized: Dict[str, float]
    """
    last_price: Optional[float] = None
    for e in entries:
        if "price" in e:
            try:
                last_price = float(e["price"])
            except (TypeError, ValueError):
                continue

    realized = 0.0
    open_pnl = 0.0
    by_strategy: Dict[str, float] = {}

    open_pos: Optional[Position] = None

    for e in entries:
        etype = e.get("type")
        strat = str(e.get("strategy") or "unknown")

        price_raw = e.get("price")
        try:
            price = float(price_raw) if price_raw is not None else None
        except (TypeError, ValueError):
            price = None

        size_raw = e.get("size", 0)
        try:
            size = float(size_raw)
        except (TypeError, ValueError):
            size = 0.0

        if etype == "open":
            # If an open already exists, ignore (paper engine is one-at-a-time).
            if open_pos is None and price is not None:
                open_pos = Position(strategy=strat, entry=price, size=size)

        elif etype == "close":
            # Realize P&L if there is a matching open.
            if open_pos is not None and price is not None:
                diff = (price - open_pos.entry) * open_pos.size
                realized += diff
                by_strategy[strat] = by_strategy.get(strat, 0.0) + diff
                open_pos = None

    # Mark-to-market for any leftover open position
    if open_pos is not None and last_price is not None:
        open_pnl = (last_price - open_pos.entry) * open_pos.size

    return realized, open_pnl, by_strategy


def _ensure_csv_header(csv_path: Path) -> None:
    """Create the snapshot CSV with header if it does not exist."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "total_realized", "total_open_pnl", "by_strategy_json"])


def write_snapshot() -> None:
    """Compute P&L and append a snapshot row to the CSV."""
    entries = _read_ledger(LEDGER_PATH)
    realized, open_pnl, by_strategy = _compute_pnl(entries)

    _ensure_csv_header(SNAPSHOT_CSV)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        ts,
        f"{realized:.2f}",
        f"{open_pnl:.2f}",
        json.dumps(by_strategy, separators=(",", ":")),
    ]

    with SNAPSHOT_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    print("\n=== PNL SNAPSHOT ===")
    print(f"Time:           {ts}")
    print(f"Realized P&L:   {realized:.2f}")
    print(f"Open P&L:       {open_pnl:.2f}")
    print(f"By strategy:    {by_strategy}")
    print(f"\nSaved -> {SNAPSHOT_CSV}")


def main() -> None:
    """CLI entrypoint to write a paper P&L snapshot row."""
    parser = argparse.ArgumentParser(description="Write a paper P&L snapshot row.")
    # (Reserved for future flags, e.g., custom ledger path.)
    parser.parse_args()
    write_snapshot()


if __name__ == "__main__":
    main()
