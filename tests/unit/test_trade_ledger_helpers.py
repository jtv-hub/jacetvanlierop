"""Unit tests for key helper behaviors in the trade ledger module."""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace

import pytest

from crypto_trading_bot.ledger import trade_ledger
from crypto_trading_bot.ledger.trade_ledger import TradeLedger, _apply_exit_slippage


@pytest.fixture(name="silence_trade_logger")
def fixture_silence_trade_logger(monkeypatch):
    """Prevent disk writes by redirecting the trade logger to an in-memory stream."""

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    original_handlers = list(trade_ledger.trade_logger.handlers)
    original_level = trade_ledger.trade_logger.level
    original_propagate = trade_ledger.trade_logger.propagate
    for existing in original_handlers:
        trade_ledger.trade_logger.removeHandler(existing)
    trade_ledger.trade_logger.addHandler(handler)
    trade_ledger.trade_logger.setLevel(logging.INFO)
    trade_ledger.trade_logger.propagate = False

    yield

    trade_ledger.trade_logger.removeHandler(handler)
    for existing in original_handlers:
        trade_ledger.trade_logger.addHandler(existing)
    trade_ledger.trade_logger.setLevel(original_level)
    trade_ledger.trade_logger.propagate = original_propagate


def test_apply_exit_slippage_noop_float():
    """Returns the exact same float value unchanged."""
    assert _apply_exit_slippage("BTC/USDC", 12345.67) == 12345.67


def test_apply_exit_slippage_casts_to_float():
    """Casts string inputs to float while preserving numeric value."""
    assert _apply_exit_slippage("ETH/USDC", "100.5") == 100.5


def test_closed_trade_populates_side_and_capital_buffer(tmp_path, monkeypatch, silence_trade_logger):
    """Closing a trade ensures mandatory metadata (side, capital_buffer) is set."""

    trades_path = tmp_path / "trades.log"
    positions_path = tmp_path / "positions.jsonl"
    trades_path.write_text(
        json.dumps(
            {
                "trade_id": "T-123",
                "timestamp": "2024-01-01T00:00:00+00:00",
                "pair": "BTC/USDC",
                "size": 1.0,
                "entry_price": 100.0,
                "status": "executed",
                "strategy": "UnitTest",
                "side": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    positions_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(trade_ledger, "TRADES_LOG_PATH", str(trades_path))
    monkeypatch.setattr(trade_ledger, "POSITIONS_PATH", str(positions_path))

    ledger = TradeLedger(position_manager=SimpleNamespace(positions={}))
    ledger.update_trade("T-123", exit_price=110.0, reason="TAKE_PROFIT")

    updated_trade = ledger.trade_index["T-123"]
    assert updated_trade["status"] == "closed"
    assert updated_trade["side"] in {"long", "short"}
    assert isinstance(updated_trade["capital_buffer"], float)


def test_reload_trades_backfills_missing_metadata(tmp_path, monkeypatch, silence_trade_logger):
    """Legacy closed trades without metadata are backfilled during reload."""

    trades_path = tmp_path / "trades.log"
    positions_path = tmp_path / "positions.jsonl"
    trades_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trade_id": "legacy-1",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "pair": "ETH/USDC",
                        "size": 2.0,
                        "entry_price": 50.0,
                        "exit_price": 60.0,
                        "status": "closed",
                        "strategy": "LegacyStrategy",
                        # missing side/capital_buffer
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    positions_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(trade_ledger, "TRADES_LOG_PATH", str(trades_path))
    monkeypatch.setattr(trade_ledger, "POSITIONS_PATH", str(positions_path))

    ledger = TradeLedger(position_manager=SimpleNamespace(positions={}))
    legacy_trade = ledger.trade_index["legacy-1"]

    assert legacy_trade["side"] in {"long", "short"}
    assert isinstance(legacy_trade["capital_buffer"], float)
