"""Unit tests for the NSGA-III healthcheck helpers."""

from __future__ import annotations

import os
import sqlite3

import pytest

from crypto_trading_bot.nsga3 import nsga3_healthcheck

DB_PATH = "db/trades.db"


def setup_module():
    """Create a lightweight trades database for the test module."""
    os.makedirs("db", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                roi REAL
            );"""
        )
        conn.commit()


def teardown_module():
    """Clean up the temporary trades database."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


def test_check_database_returns_false_on_missing_table(monkeypatch: pytest.MonkeyPatch):
    """Healthcheck should fail when the trades table is missing."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    monkeypatch.setattr(nsga3_healthcheck, "_check_logs", lambda: True)
    monkeypatch.setattr(nsga3_healthcheck, "_check_risk_guard", lambda: True)
    result = nsga3_healthcheck.run_healthcheck()
    assert result is False


def test_run_healthcheck_returns_boolean():
    """Healthcheck entrypoint should always return a boolean."""
    os.makedirs("logs", exist_ok=True)
    result = nsga3_healthcheck.run_healthcheck()
    assert isinstance(result, bool)
