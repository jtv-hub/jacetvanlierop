"""Tests for ledger balance source selection between live and paper modes."""

from __future__ import annotations

import logging

import pytest

from crypto_trading_bot.ledger import trade_ledger


class _StubPositionManager:
    def __init__(self) -> None:
        self.positions: dict[str, dict[str, float]] = {}


@pytest.fixture(autouse=True)
def _isolated_trade_environment(monkeypatch, tmp_path):
    """Ensure ledger avoids touching real logs during balance tests."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trade_ledger, "TRADES_LOG_PATH", "trades.log", raising=False)
    monkeypatch.setattr(trade_ledger, "POSITIONS_PATH", "positions.jsonl", raising=False)
    for handler in list(trade_ledger.trade_logger.handlers):
        trade_ledger.trade_logger.removeHandler(handler)
    trade_ledger.trade_logger.addHandler(logging.NullHandler())
    yield


def test_live_mode_balance_failure_raises(monkeypatch):
    """Live mode should abort when Kraken balance cannot be fetched."""

    monkeypatch.setattr(trade_ledger, "IS_LIVE", True, raising=False)

    def _kraken_fail(_asset="USDC"):
        raise RuntimeError("kraken down")

    monkeypatch.setattr(trade_ledger, "kraken_get_balance", _kraken_fail, raising=False)

    ledger = trade_ledger.TradeLedger(_StubPositionManager())

    with pytest.raises(RuntimeError, match="Failed to fetch live balance"):
        ledger.get_account_balance()


def test_paper_mode_balance_fallback(monkeypatch):
    """Paper mode should use the configured simulated balance."""

    monkeypatch.setattr(trade_ledger, "IS_LIVE", False, raising=False)

    def _kraken_should_not_run(_asset="USDC"):
        raise AssertionError("kraken_get_balance should not be called in paper mode")

    monkeypatch.setattr(trade_ledger, "kraken_get_balance", _kraken_should_not_run, raising=False)
    target_balance = 123_456.78
    monkeypatch.setitem(trade_ledger.CONFIG, "paper_mode", {"starting_balance": target_balance})

    ledger = trade_ledger.TradeLedger(_StubPositionManager())
    balance = ledger.get_account_balance()

    assert balance == pytest.approx(target_balance)
    assert ledger.get_balance_source() == "paper_simulated"
