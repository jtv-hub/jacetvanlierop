"""Persistent live-trading risk guard.

Tracks loss streaks and drawdown thresholds across process restarts so the bot
can pause live trading until an operator explicitly clears the condition.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Tuple, cast

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.config.constants import (
    DEFAULT_RISK_DRAWDOWN_THRESHOLD,
    DEFAULT_RISK_FAILURE_LIMIT,
)
from crypto_trading_bot.utils.system_logger import get_system_logger

logger = get_system_logger().getChild("risk_guard")

_STATE_CACHE: dict[str, Any] | None = None
_STATE_CACHE_MTIME: int | None = None
_ALERT_MODULE = None
_LAST_PAUSED_STATE: bool | None = None


class _SendAlertCallable(Protocol):  # pylint: disable=too-few-public-methods
    def __call__(
        self,
        message: str,
        *,
        level: str = "INFO",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Protocol describing the alerts helper signature."""


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
    global _STATE_CACHE, _STATE_CACHE_MTIME, _LAST_PAUSED_STATE  # pylint: disable=global-statement
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_paused: bool | None = None
    if _STATE_CACHE is not None:
        previous_paused = bool(_STATE_CACHE.get("paused"))
    elif _LAST_PAUSED_STATE is not None:
        previous_paused = _LAST_PAUSED_STATE
    snapshot = dict(state)
    snapshot["updated_at"] = _now()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _STATE_CACHE = dict(snapshot)
    try:
        _STATE_CACHE_MTIME = path.stat().st_mtime_ns
    except OSError:
        _STATE_CACHE_MTIME = None
    current_paused = bool(snapshot.get("paused"))
    _LAST_PAUSED_STATE = current_paused
    if previous_paused and not current_paused:
        logger.info(
            "[risk_guard] Pause cleared via state update — trading may resume",
            extra={
                "pause_reason": snapshot.get("pause_reason"),
                "pause_trigger": snapshot.get("pause_trigger"),
            },
        )
    return snapshot


def _current_state_mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def load_state(*, force_reload: bool = False) -> dict[str, Any]:
    """Return the persisted risk guard state, reloading when requested."""

    global _STATE_CACHE, _STATE_CACHE_MTIME  # pylint: disable=global-statement
    path = _state_path()
    if (
        not force_reload
        and _STATE_CACHE is not None
        and _STATE_CACHE_MTIME is not None
        and _STATE_CACHE_MTIME == _current_state_mtime(path)
    ):
        return dict(_STATE_CACHE)

    if not force_reload and _STATE_CACHE is not None and not path.exists():
        return dict(_STATE_CACHE)

    if not path.exists():
        _STATE_CACHE = _default_state()
        _STATE_CACHE_MTIME = None
        return dict(_STATE_CACHE)
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
        _STATE_CACHE = _default_state()
        _STATE_CACHE_MTIME = _current_state_mtime(path)
        return dict(_STATE_CACHE)
    state = _default_state()
    state.update(data)
    _STATE_CACHE = state
    _STATE_CACHE_MTIME = _current_state_mtime(path)
    return dict(_STATE_CACHE)


def clear_state() -> dict[str, Any]:
    """Reset the persistent risk state to defaults."""

    state = _default_state()
    snapshot = _write_state(state)
    logger.info("[risk_guard] state cleared at %s", _state_path())
    return snapshot


def state_path() -> Path:
    """Return the resolved filesystem path backing the risk state."""

    return _state_path()


def default_state() -> dict[str, Any]:
    """Return a fresh default risk guard state."""

    return _default_state()


def _failure_limit() -> int:
    live_cfg = CONFIG.get("live_mode", {}) or {}
    try:
        value = int(live_cfg.get("failure_limit", DEFAULT_RISK_FAILURE_LIMIT) or DEFAULT_RISK_FAILURE_LIMIT)
    except (TypeError, ValueError):
        value = DEFAULT_RISK_FAILURE_LIMIT
    return max(value, 1)


def _drawdown_limit() -> float:
    live_cfg = CONFIG.get("live_mode", {}) or {}
    try:
        value = float(
            live_cfg.get("drawdown_threshold", DEFAULT_RISK_DRAWDOWN_THRESHOLD) or DEFAULT_RISK_DRAWDOWN_THRESHOLD
        )
    except (TypeError, ValueError):
        value = DEFAULT_RISK_DRAWDOWN_THRESHOLD
    return max(value, 0.0)


def invalidate_cache() -> None:
    """Clear the in-memory cache so the next load reads from disk."""

    global _STATE_CACHE, _STATE_CACHE_MTIME  # pylint: disable=global-statement
    _STATE_CACHE = None
    _STATE_CACHE_MTIME = None


def is_paused(state: dict[str, Any] | None = None, *, refresh: bool = False) -> bool:
    """Return ``True`` when the guard is actively paused."""

    snapshot = state if state is not None else load_state(force_reload=refresh)
    return bool(snapshot.get("paused"))


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


def _resolve_alert_callable() -> _SendAlertCallable | None:
    """Return the alerts helper callable when available."""

    global _ALERT_MODULE  # pylint: disable=global-statement
    try:
        if _ALERT_MODULE is None:
            alerts_path = Path(__file__).resolve().parents[2] / "bot" / "utils" / "alerts.py"
            spec = importlib.util.spec_from_file_location("_alerts", alerts_path)
            if not spec or not spec.loader:
                raise ImportError("Unable to locate alerts module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[arg-type]
            _ALERT_MODULE = module
        send_alert_fn = getattr(_ALERT_MODULE, "send_alert", None)
        if callable(send_alert_fn):
            return cast(_SendAlertCallable, send_alert_fn)
    except Exception:  # pylint: disable=broad-except
        logger.debug("alerts module unavailable.", exc_info=True)
        return None
    return None


def _send_alert(
    message: str,
    *,
    level: str = "INFO",
    context: dict[str, Any] | None = None,
) -> None:
    """Send an alert if the optional alerts helper is importable."""

    send_alert_fn: _SendAlertCallable | None = _resolve_alert_callable()
    if send_alert_fn is None:
        logger.debug("Skipping alert (helper unavailable): %s", message)
        return
    send_alert_fn(message, level=level, context=context)  # pylint: disable=not-callable


def activate_pause(
    reason: str,
    *,
    trigger: str = "manual",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Force a pause and persist the state."""

    state = load_state()
    state["paused"] = True
    state["pause_reason"] = reason
    state["pause_trigger"] = trigger
    snapshot = _write_state(state)
    _send_alert(
        f"[risk_guard] Pause activated — {reason}",
        level="CRITICAL",
        context={**(context or {}), "trigger": trigger},
    )
    return snapshot


def resume_trading(
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Clear the pause flag so trading can resume."""

    state = load_state()
    state["paused"] = False
    state["pause_reason"] = None
    state["pause_trigger"] = None
    state["consecutive_failures"] = 0
    state["last_drawdown"] = 0.0
    state["max_drawdown"] = 0.0
    snapshot = _write_state(state)
    _send_alert(
        "[risk_guard] Pause cleared — trading may resume.",
        level="WARNING",
        context=context,
    )
    return snapshot


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
    if 0 < limit <= magnitude:
        already_paused = bool(state.get("paused")) and state.get("pause_trigger") == "drawdown"
        if not already_paused:
            message = (
                "[risk_guard] drawdown exceeded threshold — " f"{magnitude:.2%} ≥ {limit:.2%}; pausing live trading."
            )
            logger.warning(message)
            state["paused"] = True
            state["pause_trigger"] = "drawdown"
            state["pause_reason"] = f"drawdown {magnitude:.2%} ≥ {limit:.2%}"
            _send_alert(
                "[risk_guard] Drawdown threshold breached.",
                level="CRITICAL",
                context={"drawdown": magnitude, "limit": limit},
            )
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
                    "[risk_guard] consecutive loss limit reached (%s failures) — " "pausing live trading.",
                    streak,
                )
                if trade_id:
                    logger.debug(
                        "[risk_guard] last failure trade_id=%s exit_reason=%s",
                        trade_id,
                        exit_reason,
                    )
                state["paused"] = True
                state["pause_trigger"] = "consecutive_failures"
                state["pause_reason"] = f"{streak} consecutive losses"
                _send_alert(
                    "[risk_guard] Consecutive loss limit reached — trading paused.",
                    level="CRITICAL",
                    context={
                        "streak": streak,
                        "trade_id": trade_id,
                        "exit_reason": exit_reason,
                    },
                )
    else:
        if state.get("consecutive_failures"):
            state["consecutive_failures"] = 0
        if bool(state.get("paused")) and (state.get("pause_trigger") == "consecutive_failures"):
            state["paused"] = False
            state["pause_trigger"] = None
            state["pause_reason"] = None
            logger.info("[risk_guard] consecutive loss streak cleared — resuming live trading allowed.")

    state = _write_state(state)
    return state


__all__ = [
    "default_state",
    "activate_pause",
    "check_pause",
    "clear_state",
    "invalidate_cache",
    "is_paused",
    "load_state",
    "resume_trading",
    "state_path",
    "update_drawdown",
    "update_trade_outcome",
]
