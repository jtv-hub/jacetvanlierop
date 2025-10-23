"""Tests for reconciliation trade import and pending markers."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from crypto_trading_bot.ledger import trade_ledger as ledger_module
from scripts import audit_kraken_reconciliation as reconciliation
from scripts.audit_kraken_reconciliation import KrakenTrade, LogTrade


@pytest.fixture(autouse=True)
def _setup_tmp_ledger(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ledger_module, "TRADES_LOG_PATH", "trades.log", raising=False)
    monkeypatch.setattr(ledger_module, "POSITIONS_PATH", "positions.jsonl", raising=False)
    for handler in list(ledger_module.trade_logger.handlers):
        ledger_module.trade_logger.removeHandler(handler)
    ledger_module.trade_logger.addHandler(logging.NullHandler())
    yield


def _make_log_trade(*, trade_id: str, txid: str, minutes_ago: int = 5, pending: bool = False) -> LogTrade:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    pending_at = datetime.now(timezone.utc) if pending else None
    return LogTrade(
        trade_id=trade_id,
        timestamp=ts,
        pair="BTC/USDC",
        normalized_pair="BTC/USDC",
        size=0.01,
        side="buy",
        txids=[txid],
        reconciled=False,
        pending_reconciliation=pending,
        pending_reconciliation_at=pending_at,
        source="local",
        raw={},
    )


def _make_kraken_trade(txid: str, *, minutes_ago: int = 2, volume: float = 0.02) -> KrakenTrade:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return KrakenTrade(
        txid=txid,
        timestamp=ts,
        pair="BTC/USDC",
        normalized_pair="BTC/USDC",
        volume=volume,
        side="buy",
        raw={"price": "110", "vol": f"{volume}", "cost": "2.2", "fee": "0.002", "type": "buy"},
    )


def test_imports_missing_kraken_trade(monkeypatch):
    monkeypatch.setattr(reconciliation, "RECONCILE_IMPORT_ENABLED", True, raising=False)
    ledger = reconciliation.load_trade_ledger()

    # Seed ledger with a local trade lacking a Kraken counterpart (older than grace window)
    ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=0.01,
        strategy_name="Test",
        trade_id="local-1",
        confidence=0.9,
        entry_price=100.0,
        txid=["local-tx"],
        gross_amount=1.0,
        fee=0.001,
        net_amount=-1.001,
        balance_delta=-1.001,
    )

    log_trade = _make_log_trade(trade_id="local-1", txid="local-tx", minutes_ago=5)
    kraken_trade = _make_kraken_trade("kraken-1")

    result = reconciliation.reconcile_trades(
        matches=[],
        missing_log_trades=[log_trade],
        missing_kraken_trades=[kraken_trade],
        ledger=ledger,
        grace_seconds=60,
    )

    assert result["kraken_trades_imported"] == 1
    assert result["log_trades_marked_pending"] == 1

    imported = ledger.find_trade_by_txid("kraken-1")
    assert imported is not None
    assert imported["reconciled"] is True
    assert imported["source"] == "kraken_import"

    local_trade = ledger.trade_index["local-1"]
    assert local_trade["pending_reconciliation"] is True
    assert local_trade["pending_reconciliation_at"] is not None


def test_pending_trades_not_duplicated(monkeypatch):
    monkeypatch.setattr(reconciliation, "RECONCILE_IMPORT_ENABLED", True, raising=False)
    ledger = reconciliation.load_trade_ledger()

    log_trade = _make_log_trade(trade_id="local-1", txid="local-tx", minutes_ago=5)
    result = reconciliation.reconcile_trades(
        matches=[],
        missing_log_trades=[log_trade],
        missing_kraken_trades=[],
        ledger=ledger,
        grace_seconds=60,
    )
    assert result["log_trades_marked_pending"] == 1

    # Second run should not double-count or re-mark
    pending_log_trade = _make_log_trade(
        trade_id="local-1",
        txid="local-tx",
        minutes_ago=5,
        pending=True,
    )
    result = reconciliation.reconcile_trades(
        matches=[],
        missing_log_trades=[pending_log_trade],
        missing_kraken_trades=[],
        ledger=ledger,
        grace_seconds=60,
    )
    assert result["log_trades_marked_pending"] == 0


def test_reconciliation_idempotent_on_import(monkeypatch):
    monkeypatch.setattr(reconciliation, "RECONCILE_IMPORT_ENABLED", True, raising=False)
    ledger = reconciliation.load_trade_ledger()

    kraken_trade = _make_kraken_trade("kraken-1")
    result = reconciliation.reconcile_trades(
        matches=[],
        missing_log_trades=[],
        missing_kraken_trades=[kraken_trade],
        ledger=ledger,
        grace_seconds=60,
    )
    assert result["kraken_trades_imported"] == 1

    # Second run with the same Kraken trade should skip import
    result = reconciliation.reconcile_trades(
        matches=[],
        missing_log_trades=[],
        missing_kraken_trades=[kraken_trade],
        ledger=ledger,
        grace_seconds=60,
    )
    assert result["kraken_trades_imported"] == 0
