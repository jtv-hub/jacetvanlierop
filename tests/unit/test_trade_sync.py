from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script_module(name: str):
    module_path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


cleaner_module = _load_script_module("clean_orphaned_positions")
inspector_module = _load_script_module("inspect_trade_sync")

clean_orphaned_positions = cleaner_module.clean_orphaned_positions
inspect_trade_sync = inspector_module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")


@pytest.fixture
def seeded_logs(tmp_path: Path) -> tuple[Path, Path]:
    logs_dir = tmp_path / "logs"
    trades_path = logs_dir / "trades.log"
    positions_path = logs_dir / "positions.jsonl"

    matching_trade = {
        "trade_id": "match",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "pair": "BTC/USDC",
        "size": 0.01,
        "confidence": 0.75,
        "roi": 0.02,
        "status": "executed",
    }
    orphan_trade = {
        "trade_id": "orphan-trade",
        "timestamp": "2025-01-02T00:00:00+00:00",
        "pair": "ETH/USDC",
        "size": 0.02,
        "confidence": 0.6,
        "roi": -0.01,
        "status": "closed",
        "entry_price": 1800.0,
        "exit_price": 1750.0,
    }
    _write_jsonl(trades_path, [matching_trade, orphan_trade])

    positions = [
        {
            "trade_id": "match",
            "timestamp": "2025-01-01T00:10:00+00:00",
            "pair": "BTC/USDC",
            "size": 0.01,
        },
        {
            "trade_id": "orphan-position",
            "timestamp": "2025-01-03T00:10:00+00:00",
            "pair": "SOL/USDC",
            "size": 1.0,
        },
    ]
    _write_jsonl(positions_path, positions)
    return positions_path, trades_path


def test_inspect_and_clean_orphaned_positions(seeded_logs: tuple[Path, Path]) -> None:
    positions_path, trades_path = seeded_logs

    orphans = inspect_trade_sync.inspect_sync(positions_path, trades_path)
    assert orphans == 1

    removed = clean_orphaned_positions(positions_path, trades_path)
    assert removed == 1

    orphans_after = inspect_trade_sync.inspect_sync(positions_path, trades_path)
    assert orphans_after == 0
