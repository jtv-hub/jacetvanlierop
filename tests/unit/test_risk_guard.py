"""Tests covering the persistent risk guard state helpers."""

from __future__ import annotations

import importlib
import json
import time

import pytest

import crypto_trading_bot.safety.risk_guard as risk_guard_pkg
from crypto_trading_bot.config import CONFIG

# pylint: disable=missing-function-docstring


@pytest.fixture(name="risk_guard")
def fixture_risk_guard(tmp_path, monkeypatch):
    state_path = tmp_path / "risk_guard_state.json"
    monkeypatch.setenv("RISK_GUARD_STATE_FILE", str(state_path))

    live_cfg = CONFIG.setdefault("live_mode", {})
    original_live_cfg = dict(live_cfg)
    live_cfg.update(
        {
            "risk_state_file": str(state_path),
            "failure_limit": 5,
            "drawdown_threshold": 0.10,
        }
    )

    module = importlib.reload(risk_guard_pkg)
    yield module

    importlib.reload(risk_guard_pkg)
    monkeypatch.delenv("RISK_GUARD_STATE_FILE", raising=False)
    live_cfg.clear()
    live_cfg.update(original_live_cfg)
    if state_path.exists():
        state_path.unlink()


def test_consecutive_failures_trigger_pause(risk_guard):
    risk_guard.clear_state()
    for _ in range(5):
        state = risk_guard.update_trade_outcome(-0.01)
    paused, reason = risk_guard.check_pause(state)
    assert paused
    assert "consecutive" in (reason or "")
    assert state["pause_trigger"] == "consecutive_failures"


def test_success_resets_pause_after_losses(risk_guard):
    risk_guard.clear_state()
    for _ in range(5):
        risk_guard.update_trade_outcome(-0.02)
    paused, _ = risk_guard.check_pause()
    assert paused

    state = risk_guard.update_trade_outcome(0.05)
    paused, _ = risk_guard.check_pause(state)
    assert not paused
    assert state["consecutive_failures"] == 0
    assert state.get("pause_trigger") is None


def test_drawdown_triggers_and_persists_pause(risk_guard):
    risk_guard.clear_state()
    state = risk_guard.update_drawdown(0.12)
    paused, reason = risk_guard.check_pause(state)
    assert paused
    assert state["pause_trigger"] == "drawdown"
    assert "drawdown" in (reason or "")

    state = risk_guard.update_trade_outcome(0.08)
    paused, _ = risk_guard.check_pause(state)
    assert paused  # drawdown pause requires manual reset


def test_clear_state_resets_file(risk_guard):
    risk_guard.update_trade_outcome(-0.5)
    path = risk_guard.state_path()
    assert path.exists()

    state = risk_guard.clear_state()
    assert path.exists()
    assert state["consecutive_failures"] == 0
    assert not state["paused"]

    reloaded = risk_guard.load_state()
    assert reloaded["consecutive_failures"] == 0
    assert not reloaded["paused"]


def test_is_paused_refreshes_after_external_reset(risk_guard):
    risk_guard.clear_state()
    risk_guard.activate_pause("manual override", trigger="manual")
    assert risk_guard.is_paused()

    path = risk_guard.state_path()
    # Ensure filesystem timestamp moves forward before overwriting.
    time.sleep(0.01)
    external_state = risk_guard.default_state()
    external_state["paused"] = False
    external_state["pause_trigger"] = None
    external_state["pause_reason"] = None
    path.write_text(json.dumps(external_state), encoding="utf-8")

    assert not risk_guard.is_paused()
    # Explicit refresh flag should also re-read from disk without relying on cache.
    assert not risk_guard.is_paused(refresh=True)
