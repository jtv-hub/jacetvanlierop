"""Tests for ledger behaviour when recording Kraken fill data."""

from __future__ import annotations

import logging

import pytest

from crypto_trading_bot.ledger import trade_ledger as ledger_module
from crypto_trading_bot.ledger.trade_ledger import TradeLedger


class _PMStub:
    """Minimal position manager stub for testing."""

    def __init__(self):
        self.positions = {}


@pytest.fixture(autouse=True)
def _clean_trade_logger(monkeypatch, tmp_path):
    """Redirect ledger file paths and logger handlers during tests."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ledger_module, "TRADES_LOG_PATH", "trades.log", raising=False)
    monkeypatch.setattr(ledger_module, "POSITIONS_PATH", "positions.jsonl", raising=False)

    # Replace handlers with a null handler to avoid filesystem writes.
    for handler in list(ledger_module.trade_logger.handlers):
        ledger_module.trade_logger.removeHandler(handler)
    ledger_module.trade_logger.addHandler(logging.NullHandler())

    yield


def test_log_trade_records_fill_metadata(monkeypatch):
    """log_trade should persist Kraken txid and fill/fee details."""

    # Ensure CONFIG bounds are permissive for the test volume.
    monkeypatch.setitem(ledger_module.CONFIG, "trade_size", {"min": 0.0001, "max": 10.0})

    ledger = TradeLedger(_PMStub())

    fill_info = {
        "price": 100.0,
        "quantity": 0.01,
        "cost": 1.0,
        "fee": 0.0002,
        "type": "market",
        "time": "2025-01-01T00:00:00Z",
    }

    # When provided with Kraken fill data (txid list + fee/cost), the ledger should persist
    # the list form so reconciliation can use the full identifier set.
    trade_id = ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=0.01,
        strategy_name="UnitTestStrategy",
        confidence=0.9,
        txid=["KRK123ABC"],
        fills=[fill_info],
        gross_amount=1.0,
        fee=0.0002,
        net_amount=-1.0002,
        balance_delta=-1.0002,
        fill_price=100.0,
        filled_volume=0.01,
    )

    recorded = ledger.trade_index[trade_id]
    assert recorded["txid"] == ["KRK123ABC"]
    assert recorded["txid"][0] == "KRK123ABC"
    assert recorded["gross_amount"] == pytest.approx(1.0)
    assert recorded["fee"] == pytest.approx(0.0002)
    assert recorded["net_amount"] == pytest.approx(-1.0002)
    assert recorded["balance_delta"] == pytest.approx(-1.0002)
    assert recorded["account_balance"] == pytest.approx(-1.0002)
    assert recorded["entry_price"] == pytest.approx(100.0)
    assert recorded["cost_basis"] == pytest.approx(1.0)
    assert recorded["fills"][0]["price"] == pytest.approx(100.0)
