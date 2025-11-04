"""Persistent live-trading risk guard.

Tracks loss streaks and drawdown thresholds across process restarts so the bot
can pause live trading until an operator explicitly clears the condition.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, Tuple, cast

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.config.constants import (
    DEFAULT_RISK_DRAWDOWN_THRESHOLD,
    DEFAULT_RISK_FAILURE_LIMIT,
)
from crypto_trading_bot.utils.system_logger import get_system_logger

logger = get_system_logger().getChild("risk_guard")

PAUSED_REASON_PATH = Path("logs/paused_reason.json").resolve()
TRADES_LOG_PATH = Path("logs/trades.log").resolve()

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


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return _now()


def _start_of_next_utc_day(reference: datetime | None = None) -> str:
    """Return the ISO timestamp for the next UTC midnight following ``reference``."""

    reference = reference or datetime.now(timezone.utc)
    midnight = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    next_day = midnight + timedelta(days=1)
    return next_day.isoformat()


def _start_of_next_iso_week(reference: datetime | None = None) -> str:
    """Return the ISO timestamp for the start of the next ISO week (Monday at 00:00 UTC)."""

    reference = reference or datetime.now(timezone.utc)
    midnight = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_week = midnight - timedelta(days=midnight.weekday())
    next_week_start = start_of_week + timedelta(days=7)
    return next_week_start.isoformat()


def _parse_trade_timestamp(trade: dict[str, Any]) -> datetime | None:
    """Best-effort parsing of trade timestamps."""

    candidates = [
        trade.get("closed_at"),
        trade.get("exit_timestamp"),
        trade.get("exit_time"),
        trade.get("exit_at"),
        trade.get("timestamp"),
    ]
    for raw in candidates:
        if not isinstance(raw, str) or not raw:
            continue
        cleaned = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _aggregate_roi(values: list[float]) -> float:
    """Return the sum of ROI values, ignoring non-numeric entries."""

    total = 0.0
    for value in values:
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _daily_and_weekly_roi(trades: list[dict[str, Any]]) -> tuple[float, float]:
    """Return aggregate ROI for the current UTC day and ISO week."""

    if not trades:
        return 0.0, 0.0

    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_week = midnight - timedelta(days=midnight.weekday())

    daily_values: list[float] = []
    weekly_values: list[float] = []

    for trade in trades:
        timestamp = _parse_trade_timestamp(trade)
        if timestamp is None:
            continue
        roi_raw = trade.get("roi")
        try:
            roi_value = float(roi_raw)
        except (TypeError, ValueError):
            continue
        if timestamp >= midnight:
            daily_values.append(roi_value)
        if timestamp >= start_of_week:
            weekly_values.append(roi_value)

    daily_total = _aggregate_roi(daily_values)
    weekly_total = _aggregate_roi(weekly_values)
    return daily_total, weekly_total


def _daily_and_weekly_drawdown_pct(trades: list[dict[str, Any]]) -> tuple[float, float]:
    """Return drawdown magnitudes (percentage) for the current UTC day and ISO week."""

    daily_total, weekly_total = _daily_and_weekly_roi(trades)
    daily_drawdown_pct = abs(min(daily_total, 0.0)) * 100.0
    weekly_drawdown_pct = abs(min(weekly_total, 0.0)) * 100.0
    return daily_drawdown_pct, weekly_drawdown_pct


def _load_closed_trades() -> list[dict[str, Any]]:
    """Return closed trades recorded in the trades log."""

    if not TRADES_LOG_PATH.exists():
        return []
    trades: list[dict[str, Any]] = []
    try:
        with TRADES_LOG_PATH.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    trade = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = str(trade.get("status", "")).lower()
                if status != "closed":
                    continue
                trades.append(trade)
    except OSError as exc:
        logger.error("[risk_guard] Failed to read trades log %s: %s", TRADES_LOG_PATH, exc)
        return []
    return trades


def _write_paused_reason(reason: str, *, resume_at: str | None = None) -> None:
    """Persist the reason for auto-pausing new entries."""

    payload = {
        "reason": reason,
        "updated_at": _utc_now_iso(),
    }
    if resume_at:
        payload["resume_at"] = resume_at
    try:
        PAUSED_REASON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PAUSED_REASON_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        logger.warning("[risk_guard] Failed to persist pause reason: %s", exc)


def _clear_paused_reason_file() -> None:
    """Remove the paused reason file when trading resumes."""

    try:
        if PAUSED_REASON_PATH.exists():
            PAUSED_REASON_PATH.unlink()
    except OSError as exc:
        logger.debug("[risk_guard] Failed to clear paused_reason file: %s", exc)


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


def should_allow_new_entry(context: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    """Return ``(allow, reason)`` gating new entries based on calendar drawdowns."""

    del context  # Context hook reserved for future enhancements.
    trades = _load_closed_trades()
    auto_cfg = CONFIG.get("auto_pause", {}) or {}
    try:
        max_daily = float(auto_cfg.get("max_daily_drawdown_pct", 5.0))
    except (TypeError, ValueError):
        max_daily = 5.0
    try:
        max_weekly = float(auto_cfg.get("max_weekly_drawdown_pct", 10.0))
    except (TypeError, ValueError):
        max_weekly = 10.0
    max_daily = max(max_daily, 0.0)
    max_weekly = max(max_weekly, 0.0)

    if not trades or (max_daily <= 0 and max_weekly <= 0):
        state = load_state()
        try:
            current_drawdown = float(state.get("last_drawdown", 0.0) or 0.0)
        except (TypeError, ValueError):
            current_drawdown = 0.0
        if current_drawdown != 0.0:
            state["last_drawdown"] = 0.0
            _write_state(state)
        _clear_paused_reason_file()
        return True, None

    daily_roi, weekly_roi = _daily_and_weekly_roi(trades)
    daily_drawdown_pct, weekly_drawdown_pct = _daily_and_weekly_drawdown_pct(trades)

    state = load_state()
    worst_drawdown = min(daily_roi, weekly_roi, 0.0)
    state["last_drawdown"] = float(worst_drawdown)
    _write_state(state)

    reason: str | None = None
    resume_at: str | None = None
    if max_daily > 0 and daily_drawdown_pct >= max_daily:
        reason = f"Daily drawdown {daily_drawdown_pct:.2f}% exceeds limit {max_daily:.2f}%"
        resume_at = _start_of_next_utc_day()
    elif max_weekly > 0 and weekly_drawdown_pct >= max_weekly:
        reason = f"Weekly drawdown {weekly_drawdown_pct:.2f}% exceeds limit {max_weekly:.2f}%"
        resume_at = _start_of_next_iso_week()

    if reason:
        _write_paused_reason(reason, resume_at=resume_at)
        return False, reason

    _clear_paused_reason_file()
    return True, None


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


def trigger_panic_exit_if_needed() -> bool:
    """Force-close open positions when weekly ROI breaches the hard drawdown guard."""

    trades = _load_closed_trades()
    if not trades:
        return False

    _, weekly_roi = _daily_and_weekly_roi(trades)
    weekly_threshold = -0.15
    if weekly_roi > weekly_threshold:
        return False

    try:
        # Imported lazily to avoid circular imports during module initialization.
        from crypto_trading_bot.bot import trading_logic  # type: ignore import-not-found
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("[risk_guard] Panic exit unavailable — trading_logic import failed: %s", exc)
        return False

    position_manager = getattr(trading_logic, "position_manager", None)
    ledger = getattr(trading_logic, "ledger", None)
    if position_manager is None or ledger is None:
        logger.error("[risk_guard] Panic exit unavailable — position manager or ledger missing.")
        return False

    exits = 0
    for trade_id, position in list(position_manager.positions.items()):
        exit_price_raw = (
            position.get("current_price")
            or position.get("last_price")
            or position.get("exit_price")
            or position.get("entry_price")
        )
        try:
            exit_price = float(exit_price_raw)
        except (TypeError, ValueError):
            entry_price = position.get("entry_price")
            try:
                exit_price = float(entry_price)
            except (TypeError, ValueError):
                exit_price = 0.0
        if exit_price <= 0:
            logger.warning(
                "[risk_guard] Panic exit using fallback price for trade %s",
                trade_id,
                extra={"fallback_price": exit_price},
            )
        try:
            ledger.update_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                reason="PANIC_EXIT",
            )
            exits += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[risk_guard] Failed to update trade %s during panic exit: %s", trade_id, exc)
            continue

        if trade_id in position_manager.positions:
            del position_manager.positions[trade_id]

    if exits == 0:
        return False

    state = load_state()
    state["paused"] = True
    state["pause_trigger"] = "panic_exit"
    state["pause_reason"] = "weekly drawdown exceeded 15%"
    state["last_drawdown"] = float(min(weekly_roi, 0.0))
    _write_state(state)

    _write_paused_reason(
        "Panic exit triggered — weekly drawdown exceeded 15%",
        resume_at=_start_of_next_iso_week(),
    )

    logger.critical(
        "[risk_guard] Panic exit executed for %d positions | weekly_roi=%.2f%%",
        exits,
        weekly_roi * 100.0,
    )
    return True


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
    "should_allow_new_entry",
    "state_path",
    "trigger_panic_exit_if_needed",
    "update_drawdown",
    "update_trade_outcome",
]
