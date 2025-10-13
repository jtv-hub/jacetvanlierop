#!/usr/bin/env python3
"""Inspect and optionally repair trade/position synchronization."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from clean_orphaned_positions import clean_orphaned_positions
except ImportError:  # pragma: no cover - fallback for module execution
    import importlib.util

    helper_path = Path(__file__).resolve().parent / "clean_orphaned_positions.py"
    spec = importlib.util.spec_from_file_location("clean_orphaned_positions", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    clean_orphaned_positions = module.clean_orphaned_positions  # type: ignore[attr-defined]


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


@dataclass
class TradeSnapshot:
    trade_id: str
    timestamp: str | None
    pair: str | None
    roi: float | None
    confidence: float | None
    status: str | None

    @classmethod
    def from_trade(cls, trade: dict) -> "TradeSnapshot":
        def _float(name: str) -> float | None:
            value = trade.get(name)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            trade_id=str(trade.get("trade_id")),
            timestamp=trade.get("timestamp"),
            pair=trade.get("pair"),
            roi=_float("roi"),
            confidence=_float("confidence"),
            status=str(trade.get("status")) if trade.get("status") else None,
        )


def _summarize(snapshot: TradeSnapshot) -> str:
    return (
        f"trade_id={snapshot.trade_id} pair={snapshot.pair or '<unknown>'} "
        f"roi={snapshot.roi if snapshot.roi is not None else '<na>'} "
        f"confidence={snapshot.confidence if snapshot.confidence is not None else '<na>'} "
        f"timestamp={snapshot.timestamp or '<na>'} status={snapshot.status or '<na>'}"
    )


def _validate_closed_trade(trade: dict) -> list[str]:
    issues: list[str] = []
    if str(trade.get("status", "")).lower() != "closed":
        return issues
    if not trade.get("entry_price"):
        issues.append("missing entry_price")
    if not trade.get("exit_price"):
        issues.append("missing exit_price")
    return issues


def inspect_sync(positions_path: Path, trades_path: Path) -> int:
    trades = _load_jsonl(trades_path)
    positions = _load_jsonl(positions_path)

    trade_index = {tr.get("trade_id"): tr for tr in trades if tr.get("trade_id")}
    position_index = {pos.get("trade_id"): pos for pos in positions if pos.get("trade_id")}

    orphan_positions = [pid for pid in position_index if pid not in trade_index]
    missing_positions = [tid for tid in trade_index if tid not in position_index]

    if orphan_positions:
        print("Positions without trades:")
        for trade_id in orphan_positions:
            pos = position_index[trade_id]
            summary = {
                "trade_id": trade_id,
                "pair": pos.get("pair"),
                "size": pos.get("size"),
                "timestamp": pos.get("timestamp"),
            }
            print("  -", json.dumps(summary, separators=(",", ":")))
        print()
    else:
        print("No position orphans detected.")

    if missing_positions:
        print("Trades without positions:")
        for trade_id in missing_positions:
            snap = TradeSnapshot.from_trade(trade_index[trade_id])
            print("  -", _summarize(snap))
        print()
    else:
        print("No trades missing positions.")

    validation_issues = 0
    for trade in trades:
        problems = _validate_closed_trade(trade)
        if problems:
            validation_issues += 1
            snap = TradeSnapshot.from_trade(trade)
            print(
                "[warning] Closed trade has issues:",
                _summarize(snap),
                "issues=" + ",".join(problems),
            )

    return len(orphan_positions)


def _auto_fix(orphan_count: int, positions_path: Path, trades_path: Path) -> None:
    if not orphan_count:
        print("No fixes required; positions and trades already aligned.")
        return
    removed = clean_orphaned_positions(positions_path, trades_path)
    print(f"Removed {removed} orphaned position(s).")


def main(argv: Iterable[str] | None = None) -> None:
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
    parser.add_argument("--fix", action="store_true", help="Automatically remove orphaned positions")
    args = parser.parse_args(list(argv) if argv is not None else None)

    orphans = inspect_sync(args.positions, args.trades)
    if args.fix:
        _auto_fix(orphans, args.positions, args.trades)


if __name__ == "__main__":
    main()
