from __future__ import annotations

import importlib

import pytest

from crypto_trading_bot.config import CONFIG


@pytest.fixture
def risk_guard(tmp_path, monkeypatch):
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

    import crypto_trading_bot.safety.risk_guard as risk_guard_module

    module = importlib.reload(risk_guard_module)
    yield module

    importlib.reload(risk_guard_module)
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
