#!/usr/bin/env python3
"""Remove orphaned positions that lack matching trades."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")


def clean_orphaned_positions(positions_path: Path, trades_path: Path) -> int:
    trades = _load_jsonl(trades_path)
    trade_ids = {entry.get("trade_id") for entry in trades if entry.get("trade_id")}

    positions = _load_jsonl(positions_path)
    retained: list[dict] = []
    removed = 0
    for position in positions:
        trade_id = position.get("trade_id")
        if trade_id and trade_id in trade_ids:
            retained.append(position)
        else:
            removed += 1

    if removed:
        backup_path = positions_path.with_name("positions_backup.jsonl")
        if positions_path.exists():
            shutil.copy2(positions_path, backup_path)
        _write_jsonl(positions_path, retained)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("logs/positions.jsonl"),
        help="Path to positions jsonl file (default: logs/positions.jsonl)",
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path("logs/trades.log"),
        help="Path to trades log file (default: logs/trades.log)",
    )
    args = parser.parse_args()

    removed = clean_orphaned_positions(args.positions, args.trades)
    if removed:
        print(f"Removed {removed} orphaned position(s); backup saved alongside original file.")
    else:
        print("No orphaned positions detected.")


if __name__ == "__main__":
    main()
