"""Persistent live-trading risk guard.

Tracks loss streaks and drawdown thresholds across process restarts so the bot
can pause live trading until an operator explicitly clears the condition.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

from crypto_trading_bot.config import CONFIG

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path() -> Path:
    live_cfg = CONFIG.get("live_mode", {}) or {}
    candidate = live_cfg.get("risk_state_file") or os.getenv("RISK_GUARD_STATE_FILE")
    if not candidate:
        candidate = "logs/risk_guard_state.json"
    return Path(str(candidate)).expanduser().resolve()


def _default_state() -> dict[str, Any]:
    return {
        "consecutive_failures": 0,
        "lifetime_failures": 0,
        "paused": False,
        "pause_reason": None,
        "pause_trigger": None,
        "last_roi": None,
        "last_drawdown": 0.0,
        "max_drawdown": 0.0,
        "updated_at": _now(),
        "state_version": 1,
    }


def _write_state(state: dict[str, Any]) -> dict[str, Any]:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(state)
    snapshot["updated_at"] = _now()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return snapshot


def load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return _default_state()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Risk guard state malformed (not a mapping).")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "[risk_guard] Failed to read state at %s: %s — resetting to defaults.",
            path,
            exc,
        )
        return _default_state()
    state = _default_state()
    state.update(data)
    return state


def clear_state() -> dict[str, Any]:
    """Reset the persistent risk state to defaults."""

    state = _default_state()
    snapshot = _write_state(state)
    logger.info("[risk_guard] state cleared at %s", _state_path())
    return snapshot


def state_path() -> Path:
    """Return the resolved filesystem path backing the risk state."""

    return _state_path()


def _failure_limit() -> int:
    live_cfg = CONFIG.get("live_mode", {}) or {}
    try:
        value = int(live_cfg.get("failure_limit", 5) or 5)
    except (TypeError, ValueError):
        value = 5
    return max(value, 1)


def _drawdown_limit() -> float:
    live_cfg = CONFIG.get("live_mode", {}) or {}
    try:
        value = float(live_cfg.get("drawdown_threshold", 0.10) or 0.10)
    except (TypeError, ValueError):
        value = 0.10
    return max(value, 0.0)


def check_pause(state: dict[str, Any] | None = None) -> Tuple[bool, str | None]:
    """Return ``(paused, reason)`` for the current guard state."""

    active_state = state if state is not None else load_state()
    if bool(active_state.get("paused")):
        reason = active_state.get("pause_reason")
        if isinstance(reason, str) and reason:
            return True, reason
        trigger = active_state.get("pause_trigger")
        if trigger == "consecutive_failures":
            return True, "consecutive failure limit reached"
        if trigger == "drawdown":
            return True, "drawdown threshold exceeded"
        return True, "risk guard active"
    return False, None


def update_drawdown(drawdown_pct: float | None) -> dict[str, Any]:
    """Persist the latest drawdown metric and enforce drawdown threshold."""

    state = load_state()
    if drawdown_pct is None:
        return state

    try:
        drawdown_value = float(drawdown_pct)
    except (TypeError, ValueError):
        return state

    state["last_drawdown"] = drawdown_value
    magnitude = abs(drawdown_value)
    previous_max = float(state.get("max_drawdown", 0.0) or 0.0)
    if magnitude > previous_max:
        state["max_drawdown"] = magnitude

    limit = _drawdown_limit()
    if magnitude >= limit and limit > 0:
        already_paused = bool(state.get("paused")) and state.get("pause_trigger") == "drawdown"
        if not already_paused:
            message = (
                "[risk_guard] drawdown exceeded threshold — " f"{magnitude:.2%} ≥ {limit:.2%}; pausing live trading."
            )
            logger.warning(message)
            state["paused"] = True
            state["pause_trigger"] = "drawdown"
            state["pause_reason"] = f"drawdown {magnitude:.2%} ≥ {limit:.2%}"
        else:
            state["pause_reason"] = state.get("pause_reason") or f"drawdown {magnitude:.2%} ≥ {limit:.2%}"
    state = _write_state(state)
    return state


def update_trade_outcome(
    roi: float | None,
    *,
    trade_id: str | None = None,
    exit_reason: str | None = None,
) -> dict[str, Any]:
    """Record a trade outcome and enforce the consecutive-failure guard."""

    state = load_state()
    roi_value: float | None
    try:
        roi_value = float(roi) if roi is not None else None
    except (TypeError, ValueError):
        roi_value = None

    state["last_roi"] = roi_value
    if roi_value is None:
        return _write_state(state)

    failure_limit = _failure_limit()
    is_failure = roi_value < 0

    if is_failure:
        streak = int(state.get("consecutive_failures", 0) or 0) + 1
        state["consecutive_failures"] = streak
        state["lifetime_failures"] = int(state.get("lifetime_failures", 0) or 0) + 1
        if streak >= failure_limit:
            already_paused = bool(state.get("paused")) and state.get("pause_trigger") == "consecutive_failures"
            if not already_paused:
                logger.warning(
                    "[risk_guard] consecutive loss limit reached (%s failures) — pausing live trading.",
                    streak,
                )
                if trade_id:
                    logger.debug("[risk_guard] last failure trade_id=%s exit_reason=%s", trade_id, exit_reason)
                state["paused"] = True
                state["pause_trigger"] = "consecutive_failures"
                state["pause_reason"] = f"{streak} consecutive losses"
    else:
        if state.get("consecutive_failures"):
            state["consecutive_failures"] = 0
        if bool(state.get("paused")) and state.get("pause_trigger") == "consecutive_failures":
            state["paused"] = False
            state["pause_trigger"] = None
            state["pause_reason"] = None
            logger.info("[risk_guard] consecutive loss streak cleared — resuming live trading allowed.")

    state = _write_state(state)
    return state


__all__ = [
    "check_pause",
    "clear_state",
    "load_state",
    "state_path",
    "update_drawdown",
    "update_trade_outcome",
]
