"""Tests for ROI audit cost basis and warnings."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts import audit_roi


def _write_trade(handle, **fields):
    entry = {
        "trade_id": fields.get("trade_id", "tid"),
        "timestamp": fields.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "pair": fields.get("pair", "BTC/USDC"),
        "size": fields.get("size", 0.01),
        "strategy": "Test",
        "confidence": 0.9,
        "status": "closed",
        "capital_buffer": 0.25,
        "tax_method": "FIFO",
        "realized_gain": fields.get("realized_gain", 0.0),
        "holding_period_days": 0.1,
        "roi": fields.get("roi", 0.01),
        "reason": "take_profit",
        "regime": "test",
        "gross_amount": fields.get("gross_amount"),
        "fee": fields.get("fee"),
        "filled_volume": fields.get("filled_volume"),
    }
    handle.write(json.dumps(entry) + "\n")


def test_uniform_cost_basis_suppressed_in_live(monkeypatch, tmp_path):
    monkeypatch.setattr(audit_roi, "TRADES_PATH", tmp_path / "trades.log")
    monkeypatch.setattr(audit_roi, "OUTPUT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_roi, "IS_LIVE", True, raising=False)

    with open(audit_roi.TRADES_PATH, "w", encoding="utf-8") as handle:
        _write_trade(handle, trade_id="t1", gross_amount=1.0, fee=0.001, filled_volume=0.01)
        _write_trade(handle, trade_id="t2", gross_amount=2.0, fee=0.002, filled_volume=0.02)

    summary = audit_roi.run_audit(trades_path=str(audit_roi.TRADES_PATH), initial_balance=1000.0)

    warnings = summary.get("warnings", [])
    assert all("Uniform cost_basis" not in warning for warning in warnings)
