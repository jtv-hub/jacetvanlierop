#!/usr/bin/env python3
"""Validate consistency between trades.log and positions.jsonl."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

LOGS_DIR = Path("logs")
TRADES_LOG = LOGS_DIR / "trades.log"
POSITIONS_LOG = LOGS_DIR / "positions.jsonl"


def _load_trade_ids(path: Path) -> set[str]:
    trade_ids: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            trade_id = entry.get("trade_id") or entry.get("id")
            if isinstance(trade_id, str):
                trade_ids.add(trade_id)
    return trade_ids


def _load_positions_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(f"Missing positions file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            trade_id = entry.get("trade_id")
            if isinstance(trade_id, str):
                ids.add(trade_id)
    return ids


def _print_sample(header: str, items: Iterable[str], limit: int = 10) -> None:
    sample = list(items)[:limit]
    if sample:
        print(f"{header}: {len(sample)} shown of {len(set(items))}")
        for item in sample:
            print(f"  - {item}")


def main() -> int:
    try:
        trade_ids = _load_trade_ids(TRADES_LOG)
        position_ids = _load_positions_ids(POSITIONS_LOG)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    orphans = sorted(position_ids - trade_ids)
    unused = sorted(trade_ids - position_ids)

    print(f"Trades in log: {len(trade_ids)}")
    print(f"Positions in file: {len(position_ids)}")
    print(f"Orphan positions (missing trades): {len(orphans)}")
    print(f"Unused trade entries (no position): {len(unused)}")

    if orphans:
        _print_sample("Sample orphan trades", orphans)

    if unused:
        _print_sample("Sample unused trade entries", unused)

    return 1 if orphans else 0


if __name__ == "__main__":
    sys.exit(main())
