#!/usr/bin/env python3
"""Clean orphan positions that have no corresponding trade entries."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

LOGS_DIR = Path("logs")
TRADES_LOG = LOGS_DIR / "trades.log"
POSITIONS_LOG = LOGS_DIR / "positions.jsonl"


def _load_trade_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")

    trade_ids: set[str] = set()
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


def _iter_positions(path: Path) -> Iterable[dict]:
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
            yield entry


def _backup_positions(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> int:
    try:
        trade_ids = _load_trade_ids(TRADES_LOG)
        positions_iter = list(_iter_positions(POSITIONS_LOG))
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    kept = []
    removed = []
    for entry in positions_iter:
        trade_id = entry.get("trade_id")
        if isinstance(trade_id, str) and trade_id in trade_ids:
            kept.append(entry)
        else:
            removed.append(entry)

    print(f"Total positions: {len(positions_iter)}")
    print(f"Kept positions: {len(kept)}")
    print(f"Removed orphan positions: {len(removed)}")

    if not removed:
        print("No orphan positions detected; positions.jsonl unchanged.")
        return 0

    backup_path = _backup_positions(POSITIONS_LOG)
    print(f"Backup written to {backup_path}")

    with POSITIONS_LOG.open("w", encoding="utf-8") as handle:
        for entry in kept:
            json.dump(entry, handle)
            handle.write("\n")

    print("positions.jsonl replaced with cleaned entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
