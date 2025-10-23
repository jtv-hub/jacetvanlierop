"""Tests for balance tolerance handling in audit ROI module."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts import audit_roi


def _write_trade(path, trade_id: str, gross_amount: float, fee: float, volume: float) -> None:
    entry = {
        "trade_id": trade_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": "BTC/USDC",
        "size": volume,
        "strategy": "Test",
        "confidence": 0.9,
        "status": "closed",
        "capital_buffer": 0.25,
        "tax_method": "FIFO",
        "roi": 0.01,
        "reason": "take_profit",
        "regime": "test",
        "gross_amount": gross_amount,
        "fee": fee,
        "filled_volume": volume,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def test_balance_tolerance_warning_in_live_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit_roi, "IS_LIVE", True, raising=False)
    monkeypatch.setattr(audit_roi, "BALANCE_TOLERANCE", 0.01, raising=False)
    monkeypatch.setattr(audit_roi, "TRADES_PATH", "logs/trades.log", raising=False)
    monkeypatch.setattr(audit_roi, "OUTPUT_PATH", "logs/audit.jsonl", raising=False)

    monkeypatch.setattr(audit_roi, "_fetch_balances", lambda: (1000.0, 1000.005))

    os_logs = tmp_path / "logs"
    os_logs.mkdir()
    trades_path = os_logs / "trades.log"
    trades_path.write_text("", encoding="utf-8")

    summary = audit_roi.run_audit(trades_path=str(trades_path), initial_balance=1000.0)

    assert summary["balance_within_tolerance"] is True
    assert any("balance" in warning.lower() for warning in summary["warnings"])


def test_uniform_cost_basis_suppressed_live(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit_roi, "IS_LIVE", True, raising=False)
    monkeypatch.setattr(audit_roi, "BALANCE_TOLERANCE", 0.01, raising=False)
    monkeypatch.setattr(audit_roi, "TRADES_PATH", "logs/trades.log", raising=False)
    monkeypatch.setattr(audit_roi, "OUTPUT_PATH", "logs/audit.jsonl", raising=False)
    monkeypatch.setattr(audit_roi, "_fetch_balances", lambda: (None, None))

    os_logs = tmp_path / "logs"
    os_logs.mkdir()
    trades_path = os_logs / "trades.log"
    trades_path.write_text("", encoding="utf-8")

    _write_trade(trades_path, "t1", gross_amount=1.0, fee=0.001, volume=0.01)
    _write_trade(trades_path, "t2", gross_amount=2.0, fee=0.002, volume=0.02)

    summary = audit_roi.run_audit(trades_path=str(trades_path), initial_balance=1000.0)

    warnings = summary.get("warnings", [])
    assert all("uniform cost_basis" not in warning for warning in warnings)
