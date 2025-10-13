"""Pytest smoke tests for anomaly logging and rotation settings.

This suite verifies that anomaly logs are emitted as compact JSONL and that the
shared rotating logger is configured correctly. Lightweight stubs are provided
to avoid external dependencies in minimal CI environments.
"""

import json
import math
import os
import sys
import types
import uuid
from datetime import datetime, timezone

import pytest

from src.crypto_trading_bot.bot.utils.log_rotation import get_anomalies_logger
from src.crypto_trading_bot.indicators import rsi
from src.crypto_trading_bot.learning.confidence_audit import log_anomaly
from src.crypto_trading_bot.ledger.trade_ledger import TradeLedger


def _stub_numpy_and_dotenv():
    """Provide minimal stubs for numpy and python-dotenv if missing."""
    # Stub numpy
    if "numpy" not in sys.modules:
        np_mod = types.ModuleType("numpy")

        class Arr(list):
            """Tiny list subclass with numpy-like helpers used by RSI."""

            def tolist(self):
                """Return a shallow Python list copy of this array."""
                return list(self)

        def asarray(x, dtype=float):
            # Keep signature; avoid unused-argument warning by touching dtype
            _ = dtype  # noqa: F841
            try:
                return Arr([float(v) for v in x])
            except (ValueError, TypeError):
                return Arr(list(x))

        def diff(arr):
            return Arr([arr[i + 1] - arr[i] for i in range(len(arr) - 1)])

        def where(cond, x, y):
            out = Arr()
            for i, c in enumerate(cond):
                out.append(x[i] if c else (y[i] if isinstance(y, (list, tuple, Arr)) else y))
            return out

        def isfinite(x):
            try:
                return math.isfinite(x)
            except (TypeError, ValueError):
                return False

        np_mod.asarray = asarray
        np_mod.diff = diff
        np_mod.where = where
        np_mod.isfinite = isfinite
        sys.modules["numpy"] = np_mod

    # Stub dotenv
    if "dotenv" not in sys.modules:
        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = lambda *a, **k: None
        sys.modules["dotenv"] = dotenv_mod


def _read_anomalies_lines_tail(limit=300):
    """Read the last `limit` lines of the anomalies.log file."""
    path = os.path.join("logs", "anomalies.log")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [ln.rstrip("\n") for ln in lines[-limit:]]


_stub_numpy_and_dotenv()


def test_anomaly_logging_rsi_and_audit():
    """Trigger anomaly logs from RSI and confidence_audit; validate JSONL entries."""
    marker = f"pytest-{uuid.uuid4()}"

    # Log a test anomaly via the shared anomalies logger
    rsi.anomalies_logger.info(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "test_rsi_anomaly",
                "marker": marker,
            },
            separators=(",", ":"),
        )
    )

    log_anomaly({"type": "test_audit_anomaly", "marker": marker}, source="pytest")

    # Read anomalies.log and filter entries for our marker
    lines = _read_anomalies_lines_tail()
    assert lines, "anomalies.log not found or empty"
    test_lines = [ln for ln in lines if marker in ln]
    assert len(test_lines) >= 2, "Expected at least two lines containing the marker"

    # Validate each test line is compact JSONL, with required keys, and no None values
    for ln in test_lines:
        assert "\n" not in ln, "Line should not contain newline characters"
        obj = json.loads(ln)
        assert isinstance(obj, dict)
        has_marker = obj.get("marker") == marker
        has_type = "type" in obj
        assert has_marker or has_type, "Missing marker/type"
        # Ensure none of the values are None for our marker-tagged entries
        assert all(v is not None for v in obj.values()), f"Found None in entry: {obj}"


def test_trade_ledger_schema_error_logs():
    """Force a schema error and confirm it logs an anomaly (valid JSONL)."""

    # Minimal position manager stub
    class PM:
        """Stub PositionManager for schema error logging test."""

        def __init__(self):
            self.positions = {}

    tl = TradeLedger(PM())

    # Intentionally pass wrong type for regime to trigger schema validation error
    with pytest.raises(TypeError):
        tl.log_trade("BTC/USDC", 0.01, "PytestStrategy", confidence=0.5, regime=123)

    # Find a recent schema error line
    lines = _read_anomalies_lines_tail()
    assert lines, "anomalies.log not found or empty"
    schema_lines = [ln for ln in reversed(lines) if '"type":"Trade Schema Error"' in ln]
    assert schema_lines, "No 'Trade Schema Error' entries found"
    entry = json.loads(schema_lines[0])
    assert isinstance(entry, dict)
    assert entry.get("type") == "Trade Schema Error"
    # Compact JSONL: single line, parsed successfully
    assert "\n" not in schema_lines[0]


def test_rotation_settings():
    """Confirm anomalies logger rotation settings are 10MB and 3 backups."""
    logger = get_anomalies_logger()
    assert logger.handlers, "Anomalies logger should have at least one handler"
    h = logger.handlers[0]
    assert getattr(h, "maxBytes", None) == 10 * 1024 * 1024, "maxBytes should be 10MB"
    assert getattr(h, "backupCount", None) == 3, "backupCount should be 3"


def test_missing_exit_price_anomaly_logged():
    """Calling update_trade with exit_price=None logs a 'Missing Exit Price' anomaly."""

    # Minimal stub PM
    class PM:
        """Minimal position manager stub used for tests."""

        def __init__(self):
            self.positions = {}

    tl = TradeLedger(PM())

    # First log a valid trade to obtain an id
    tid = tl.log_trade(
        trading_pair="BTC/USDC",
        trade_size=0.001,
        strategy_name="TestStrat",
        confidence=0.5,
        entry_price=20000.0,
    )

    # Now attempt update with a None exit_price — should not raise; should log anomaly
    tl.update_trade(trade_id=tid, exit_price=None, reason="pytest-missing-exit")

    lines = _read_anomalies_lines_tail()
    assert lines, "anomalies.log not found or empty"
    hits = [ln for ln in reversed(lines) if '"type":"Missing Exit Price"' in ln and f'"trade_id":"{tid}"' in ln]
    assert hits, "Missing Exit Price anomaly not found for the trade"
    obj = json.loads(hits[0])
    assert obj.get("type") == "Missing Exit Price"
    assert obj.get("trade_id") == tid
    assert obj.get("reason")
