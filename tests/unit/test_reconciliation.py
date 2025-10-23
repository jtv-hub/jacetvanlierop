"""Unit tests for reconciliation helpers in audit_kraken_reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts import audit_kraken_reconciliation as reconciliation


def _make_log_trade(**overrides):
    defaults = {
        "trade_id": "local-1",
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "pair": "BTC/USDC",
        "normalized_pair": "BTC/USDC",
        "size": 0.01,
        "side": "long",
        "txids": ["abc123"],
        "reconciled": False,
        "pending_reconciliation": False,
        "pending_reconciliation_at": None,
        "source": "local",
        "raw": {},
    }
    defaults.update(overrides)
    return reconciliation.LogTrade(**defaults)


def _make_kraken_trade(**overrides):
    defaults = {
        "txid": "abc123",
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "pair": "XBT/USDC",
        "normalized_pair": "BTC/USDC",
        "volume": 0.01,
        "side": "buy",
        "raw": {},
    }
    defaults.update(overrides)
    return reconciliation.KrakenTrade(**defaults)


def test_match_trades_exact_txid():
    log_trade = _make_log_trade()
    kraken_trade = _make_kraken_trade()

    matches, missing_logs, missing_kraken = reconciliation.match_trades(
        [log_trade],
        [kraken_trade],
    )

    assert len(matches) == 1
    assert not missing_logs
    assert not missing_kraken


def test_match_trades_within_tolerances():
    log_trade = _make_log_trade(
        pair="BTC/USDC",
        normalized_pair="BTC/USDC",
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        size=0.01,
        side="long",
        txids=[],
    )
    kraken_trade = _make_kraken_trade(
        pair="XBTUSDC",
        normalized_pair="BTC/USDC",
        timestamp=datetime(2024, 1, 1, 12, 0, 4, tzinfo=timezone.utc),
        volume=0.01008,
        side="buy",
    )

    matches, missing_logs, missing_kraken = reconciliation.match_trades(
        [log_trade],
        [kraken_trade],
    )

    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(4.0, abs=1e-6)
    assert not missing_logs
    assert not missing_kraken


def test_match_trades_exceeds_time_tolerance():
    log_trade = _make_log_trade(
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        size=0.01,
        txids=[],
    )
    kraken_trade = _make_kraken_trade(
        timestamp=datetime(2024, 1, 1, 12, 0, 6, tzinfo=timezone.utc),
        volume=0.01,
    )

    matches, missing_logs, missing_kraken = reconciliation.match_trades(
        [log_trade],
        [kraken_trade],
    )

    assert not matches
    assert missing_logs == [log_trade]
    assert missing_kraken == [kraken_trade]


def test_match_trades_exceeds_volume_tolerance():
    log_trade = _make_log_trade(
        size=0.01,
        txids=[],
    )
    kraken_trade = _make_kraken_trade(
        volume=0.0102,
        timestamp=log_trade.timestamp,
    )

    matches, missing_logs, missing_kraken = reconciliation.match_trades(
        [log_trade],
        [kraken_trade],
    )

    assert not matches
    assert missing_logs == [log_trade]
    assert missing_kraken == [kraken_trade]


def test_diagnose_log_mismatch_volume():
    log_trade = _make_log_trade(size=1.0, txids=[])
    kraken_trades = [_make_kraken_trade(volume=0.1)]

    reason = reconciliation._diagnose_log_mismatch(  # type: ignore[attr-defined]
        log_trade,
        kraken_trades,
        time_tolerance=reconciliation.TIME_TOLERANCE_SECONDS,
        volume_tolerance=reconciliation.VOLUME_TOLERANCE,
    )

    assert reason.startswith("volume_gap")


def test_diagnose_log_mismatch_timestamp():
    log_trade = _make_log_trade(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        txids=[],
    )
    kraken_trades = [_make_kraken_trade(timestamp=datetime(2024, 1, 1, 4, tzinfo=timezone.utc))]

    reason = reconciliation._diagnose_log_mismatch(  # type: ignore[attr-defined]
        log_trade,
        kraken_trades,
        time_tolerance=10,
        volume_tolerance=reconciliation.VOLUME_TOLERANCE,
    )

    assert reason.startswith("timestamp_delta")


def test_partition_legacy_unmatched(monkeypatch):
    trade_recent = _make_log_trade(
        timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
        txids=[],
    )
    trade_old = _make_log_trade(
        trade_id="legacy",
        timestamp=datetime.now(timezone.utc) - timedelta(hours=48),
        txids=[],
    )

    monkeypatch.setattr(reconciliation, "LEGACY_MISMATCH_HOURS", 24)

    legacy, active = reconciliation._partition_legacy_unmatched([trade_recent, trade_old])  # type: ignore[attr-defined]

    assert trade_old in legacy
    assert trade_recent in active
