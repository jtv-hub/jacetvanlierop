"""Persistent portfolio state helpers used for daily compounding.

This module keeps track of available trading capital based on realized
trades, derives a reinvestment rate, and stores the latest portfolio
snapshot so other services (scheduler, trading logic, reporting) can
pull a consistent view of capital.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from crypto_trading_bot.config import CONFIG, get_mode_label, is_live
from crypto_trading_bot.market_data import BalanceFetchError, get_account_balance
from crypto_trading_bot.utils.reinvestment import calculate_reinvestment_rate

logger = logging.getLogger(__name__)

STATE_FILE = "data/portfolio_state.json"
TRADES_LOG_PATH = "logs/trades.log"
DEFAULT_STARTING_BALANCE = float(CONFIG.get("paper_mode", {}).get("starting_balance", 100_000.0))
DEFAULT_REINVESTMENT_RATE = float(CONFIG.get("portfolio", {}).get("default_reinvestment_rate", 0.5))
_last_logged_balance: Tuple[Optional[float], Optional[str]] = (None, None)


def _format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def _log_detected_balance(amount: float, source: str) -> None:
    global _last_logged_balance  # noqa: PLW0603  # pylint: disable=global-statement

    last_amount, last_source = _last_logged_balance
    if last_amount == amount and last_source == source:
        return

    logger.info(
        "Detected capital (%s): %s (source=%s)",
        get_mode_label(),
        _format_currency(amount),
        source,
    )
    _last_logged_balance = (amount, source)


def _resolve_live_balance() -> Tuple[Optional[float], str]:
    """Fetch live account balance if available."""

    try:
        balance = get_account_balance(use_mock_for_paper=False)
    except BalanceFetchError as exc:
        logger.warning("Live balance fetch failed: %s", exc)
        return None, "simulated"
    if balance is None:
        return None, "simulated"

    try:
        value = float(balance)
    except (TypeError, ValueError):
        logger.warning("Received non-numeric balance from get_account_balance: %s", balance)
        return None, "simulated"

    if value <= 0:
        logger.warning("Live balance %.2f is non-positive; ignoring.", value)
        return None, "simulated"

    return value, "live_account"


def _parse_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp into a timezone-aware datetime."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _clamp_rate(candidate: Any, fallback: float) -> float:
    """Return ``candidate`` clamped to [0.0, 1.0], falling back when invalid."""
    try:
        rate = float(candidate)
        if not math.isfinite(rate):
            raise ValueError
    except (TypeError, ValueError):
        rate = float(fallback)
    return min(max(rate, 0.0), 1.0)


def _extract_trade_profit(trade: dict) -> tuple[float, float]:
    """Return (profit_amount, roi_value) for a closed trade."""
    try:
        roi_value = float(trade.get("roi", 0.0) or 0.0)
    except (TypeError, ValueError):
        roi_value = 0.0

    realized_gain = trade.get("realized_gain")
    if isinstance(realized_gain, (int, float)) and math.isfinite(realized_gain):
        return float(realized_gain), roi_value

    cost_basis = trade.get("cost_basis")
    if isinstance(cost_basis, (int, float)) and math.isfinite(cost_basis):
        return float(cost_basis) * roi_value, roi_value

    size = trade.get("size")
    entry_price = trade.get("entry_price")
    try:
        if isinstance(size, (int, float)) and isinstance(entry_price, (int, float)):
            notional = float(size) * float(entry_price)
            if math.isfinite(notional):
                return notional * roi_value, roi_value
    except (TypeError, ValueError):
        pass

    return 0.0, roi_value


def _compute_daily_reinvestment(
    trades: List[dict],
    starting_balance: float,
    *,
    previous_rate: float | None = None,
) -> dict[str, Any]:
    """Apply daily compounding with reinvestment controls to closed trades."""

    default_rate = _clamp_rate(previous_rate, DEFAULT_REINVESTMENT_RATE)
    if not trades:
        return {
            "balance": round(float(starting_balance), 2),
            "total_profit": 0.0,
            "total_reinvested": 0.0,
            "retained_profit": 0.0,
            "last_rate": default_rate,
            "daily_breakdown": [],
            "last_regime": "unknown",
        }

    # Preprocess trades with timestamps and profit information
    prepared: List[dict[str, Any]] = []
    for trade in trades:
        ts = _parse_timestamp(trade.get("timestamp"))
        profit, roi_value = _extract_trade_profit(trade)
        regime = str(trade.get("regime") or "unknown")
        prepared.append(
            {
                "timestamp": ts,
                "date": ts.date().isoformat() if ts else "unknown",
                "profit": profit,
                "roi": roi_value,
                "regime": regime,
                "override_rate": trade.get("reinvestment_rate"),
            }
        )

    prepared.sort(key=lambda item: item["timestamp"] or datetime.min.replace(tzinfo=timezone.utc))

    grouped_entries: List[dict[str, Any]] = []
    for date_key, group in itertools.groupby(prepared, key=lambda item: item["date"]):
        total_profit = 0.0
        total_roi = 0.0
        regime = "unknown"
        override_rate = None
        last_ts: datetime | None = None
        for item in group:
            total_profit += float(item["profit"])
            total_roi += float(item["roi"])
            last_ts = item["timestamp"] or last_ts
            if item["override_rate"] is not None and override_rate is None:
                override_rate = item["override_rate"]
            if regime == "unknown" and item["regime"]:
                regime = item["regime"]
        grouped_entries.append(
            {
                "date": date_key,
                "profit": total_profit,
                "roi": total_roi,
                "regime": regime,
                "override_rate": override_rate,
                "timestamp": last_ts,
            }
        )

    grouped_entries.sort(key=lambda item: item["timestamp"] or datetime.min.replace(tzinfo=timezone.utc))

    balance = float(starting_balance)
    total_profit = 0.0
    total_reinvested = 0.0
    retained_profit = 0.0
    daily_breakdown: List[dict[str, Any]] = []
    last_rate = default_rate
    last_regime = "unknown"

    for entry in grouped_entries:
        regime = entry["regime"] or "unknown"
        last_regime = regime if regime != "unknown" else last_regime
        rate = _resolve_reinvestment_rate(
            balance,
            regime,
            entry["override_rate"],
            last_rate,
        )
        last_rate = rate

        profit = float(entry["profit"])
        roi_sum = float(entry["roi"])
        if math.isclose(profit, 0.0, abs_tol=1e-9) and not math.isclose(roi_sum, 0.0, abs_tol=1e-9):
            profit = balance * roi_sum
        total_profit += profit
        if profit >= 0:
            reinvestment = profit * rate
            retained_profit += profit - reinvestment
        else:
            reinvestment = profit  # apply full loss impact

        balance += reinvestment
        total_reinvested += reinvestment
        daily_breakdown.append(
            {
                "date": entry["date"],
                "regime": regime,
                "profit": round(profit, 6),
                "reinvestment_rate": round(rate, 6),
                "reinvested": round(reinvestment, 6),
                "retained": round((profit - reinvestment) if profit >= 0 else 0.0, 6),
            }
        )

    return {
        "balance": round(balance, 2),
        "total_profit": round(total_profit, 6),
        "total_reinvested": round(total_reinvested, 6),
        "retained_profit": round(retained_profit, 6),
        "last_rate": last_rate,
        "daily_breakdown": daily_breakdown,
        "last_regime": last_regime,
    }


def _resolve_reinvestment_rate(
    balance: float,
    regime: str,
    override_rate: Any,
    fallback: float,
) -> float:
    """Determine the reinvestment rate for a given balance/regime snapshot."""
    if override_rate is not None:
        return _clamp_rate(override_rate, fallback)

    try:
        dynamic = calculate_reinvestment_rate(balance, regime)
    except Exception:  # pylint: disable=broad-exception-caught
        dynamic = None

    if dynamic is None:
        return _clamp_rate(None, fallback)

    return _clamp_rate(dynamic, fallback)


def load_closed_trades(log_path: str = TRADES_LOG_PATH) -> List[dict]:
    """Return closed trades with valid ROI information from the trade log."""
    if not os.path.exists(log_path):
        return []

    closed: List[dict] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                trade = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = (trade.get("status") or "").lower()
            roi = trade.get("roi")
            size = trade.get("size")
            if status == "closed" and isinstance(roi, (int, float)) and isinstance(size, (int, float)):
                closed.append(trade)
    return closed


def _extract_rois(closed_trades: List[dict]) -> List[float]:
    values: List[float] = []
    for trade in closed_trades:
        try:
            values.append(float(trade.get("roi")))
        except (TypeError, ValueError):
            continue
    return values


def _infer_regime_from_trades(closed_trades: List[dict]) -> str:
    """Derive a lightweight market regime signal from recent trade outcomes."""
    rois = _extract_rois(closed_trades)
    if len(rois) < 5:
        return "unknown"

    wins = sum(1 for roi in rois if roi > 0)
    win_rate = wins / len(rois)
    avg_roi = mean(rois)
    dispersion = pstdev(rois) if len(rois) > 1 else 0.0

    if avg_roi > 0.002 and win_rate >= 0.6 and dispersion < 0.02:
        return "trending"
    if avg_roi < -0.002 and win_rate < 0.45:
        return "chop" if dispersion < 0.02 else "volatile"
    if dispersion >= 0.02:
        return "volatile"
    if abs(avg_roi) < 0.0005 and dispersion < 0.005:
        return "flat"
    return "chop"


def _clamp_buffer(value: float) -> float:
    return max(0.15, min(float(value), 1.0))


def _build_regime_buffer_profile(base_buffer: float, reinvestment_rate: float) -> Dict[str, float]:
    """Scale the dynamic buffer across regimes for downstream consumers."""

    trend_multiplier = max(0.6, 1.0 - 0.4 * reinvestment_rate)
    volatile_multiplier = 1.2 + 0.3 * (1.0 - reinvestment_rate)
    chop_multiplier = 1.0 + 0.15 * (1.0 - reinvestment_rate)
    flat_multiplier = max(0.7, 0.9 - 0.2 * reinvestment_rate)

    profile = {
        "trending": _clamp_buffer(base_buffer * trend_multiplier),
        "chop": _clamp_buffer(base_buffer * chop_multiplier),
        "volatile": _clamp_buffer(base_buffer * volatile_multiplier),
        "flat": _clamp_buffer(base_buffer * flat_multiplier),
        "unknown": _clamp_buffer(base_buffer),
    }
    return profile


def _build_composite_buffer_profile(
    regime_profile: Dict[str, float],
    reinvestment_rate: float,
) -> Dict[str, float]:
    """Special-case buffer targets for the CompositeStrategy."""

    composite_profile: Dict[str, float] = {}
    for regime, base_value in regime_profile.items():
        if regime == "trending":
            multiplier = max(0.55, 0.85 - 0.25 * reinvestment_rate)
        elif regime == "volatile":
            multiplier = 1.25 + 0.2 * (1.0 - reinvestment_rate)
        elif regime == "chop":
            multiplier = 1.1 + 0.1 * (1.0 - reinvestment_rate)
        elif regime == "flat":
            multiplier = max(0.7, 0.95 - 0.1 * reinvestment_rate)
        else:
            multiplier = 1.0
        composite_profile[regime] = _clamp_buffer(base_value * multiplier)
    return composite_profile


def load_state() -> Dict[str, Any]:
    """Read the persisted portfolio state from disk."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    """Persist the portfolio state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=4)


def get_current_market_regime() -> str:
    """Placeholder hook for regime detection integration."""
    # NOTE: replace with real regime detection once available
    return "trending"


def refresh_portfolio_state(
    *,
    log_path: str = TRADES_LOG_PATH,
    starting_balance: Optional[float] = None,
) -> Dict[str, Any]:
    """Recompute and persist the latest portfolio snapshot."""
    prev_state = load_state()
    closed_trades = load_closed_trades(log_path)
    if starting_balance is not None:
        effective_start = float(starting_balance)
    else:
        effective_start = DEFAULT_STARTING_BALANCE
    previous_rate = None
    if prev_state:
        try:
            previous_rate = float(prev_state.get("reinvestment_rate", DEFAULT_REINVESTMENT_RATE))
        except (TypeError, ValueError):
            previous_rate = None
    if not closed_trades:
        previous_rate = None

    compounding = _compute_daily_reinvestment(
        closed_trades,
        effective_start,
        previous_rate=previous_rate,
    )
    available_capital = compounding["balance"]
    reinvestment_rate = compounding["last_rate"]
    reinvestment_log = compounding["daily_breakdown"]
    total_profit = compounding["total_profit"]
    total_reinvested = compounding["total_reinvested"]
    retained_profit = compounding["retained_profit"]
    recent_reinvest_regime = compounding["last_regime"]
    capital_source = "paper_compounded"

    if is_live:
        live_balance, source = _resolve_live_balance()
        if live_balance is not None:
            available_capital = live_balance
            capital_source = source
            reinvestment_rate = _resolve_reinvestment_rate(
                available_capital,
                recent_reinvest_regime,
                None,
                reinvestment_rate,
            )
        else:
            logger.debug("Live balance unavailable; falling back to simulated capital.")

    _log_detected_balance(available_capital, capital_source)

    prev_baseline = 0.0
    if prev_state:
        try:
            prev_baseline = float(prev_state.get("baseline_capital", 0.0) or 0.0)
        except (TypeError, ValueError):
            prev_baseline = 0.0
    if not closed_trades:
        prev_baseline = effective_start

    try:
        prev_all_time_high = float(prev_state.get("all_time_high", prev_state.get("peak_capital", 0.0)) or 0.0)
    except (TypeError, ValueError):
        prev_all_time_high = 0.0
    if not closed_trades:
        prev_all_time_high = available_capital

    all_time_high = prev_all_time_high if prev_all_time_high > 0 else available_capital
    if available_capital > all_time_high:
        all_time_high = available_capital

    peak_capital = all_time_high

    if capital_source == "live_account":
        if prev_state.get("capital_source") != "live_account" or prev_baseline <= 0:
            baseline_capital = available_capital
        else:
            baseline_capital = prev_baseline
    else:
        baseline_capital = prev_baseline if prev_baseline > 0 else effective_start

    raw_drawdown = 0.0
    if all_time_high > 0:
        raw_drawdown = (all_time_high - available_capital) / all_time_high
    drawdown_pct = min(max(raw_drawdown, 0.0), 1.0)

    raw_drawdown_limit = os.getenv("MAX_DRAWDOWN_LIMIT")
    cfg_drawdown_limit = float(CONFIG.get("auto_pause", {}).get("max_drawdown_pct", 0.10) or 0.0)
    drawdown_limit = cfg_drawdown_limit
    if raw_drawdown_limit is not None:
        try:
            drawdown_limit = float(raw_drawdown_limit)
        except ValueError:
            logger.warning(
                "Invalid MAX_DRAWDOWN_LIMIT env value %s; falling back to %.2f%%",
                raw_drawdown_limit,
                cfg_drawdown_limit * 100,
            )
    drawdown_limit = min(max(drawdown_limit, 0.0), 1.0)

    logger.debug(
        "[DRAWDOWN] Balance=%s | High=%s | Drawdown=%.2f%% | Limit=%.2f%%",
        _format_currency(available_capital),
        _format_currency(all_time_high),
        drawdown_pct * 100,
        drawdown_limit * 100,
    )

    total_roi = 0.0
    if baseline_capital > 0:
        total_roi = (available_capital / baseline_capital) - 1.0

    most_recent_close: datetime | None = None
    for trade in closed_trades:
        ts = _parse_timestamp(trade.get("timestamp"))
        if ts is None:
            continue
        if most_recent_close is None or ts > most_recent_close:
            most_recent_close = ts

    inferred_regime_from_trades = _infer_regime_from_trades(closed_trades)

    dynamic_buffer = 0.35
    try:
        # Imported lazily to avoid circular dependency during module import.
        from crypto_trading_bot.risk import risk_manager  # pylint: disable=import-outside-toplevel

        candidate = risk_manager.get_dynamic_buffer()
    except ImportError:  # pragma: no cover - diagnostics only
        candidate = None
    else:
        try:
            dynamic_buffer = float(candidate)
        except (TypeError, ValueError):
            dynamic_buffer = 0.35

    regime_candidates = [
        inferred_regime_from_trades,
        recent_reinvest_regime,
        get_current_market_regime(),
    ]
    regime = next((r for r in regime_candidates if r and r != "unknown"), "unknown")
    reinvestment_rate = _clamp_rate(reinvestment_rate, DEFAULT_REINVESTMENT_RATE)
    if not closed_trades:
        reinvestment_rate = DEFAULT_REINVESTMENT_RATE
    regime_buffers = _build_regime_buffer_profile(dynamic_buffer, reinvestment_rate)
    composite_buffers = _build_composite_buffer_profile(regime_buffers, reinvestment_rate)

    snapshot: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_capital": round(float(available_capital), 2),
        "reinvestment_rate": float(reinvestment_rate),
        "last_close_date": (most_recent_close.date().isoformat() if most_recent_close else None),
        "market_regime": regime,
        "capital_buffer": dynamic_buffer,
        "regime_capital_buffers": regime_buffers,
        "strategy_buffers": {"CompositeStrategy": composite_buffers},
        "active_composite_buffer": composite_buffers.get(
            regime,
            regime_buffers.get(regime, dynamic_buffer),
        ),
        "daily_reinvestment": reinvestment_log,
        "total_reinvested": float(total_reinvested),
        "retained_profit": float(retained_profit),
        "total_realized_profit": float(total_profit),
        "starting_balance": float(baseline_capital if baseline_capital > 0 else effective_start),
        "closed_trade_count": len(closed_trades),
        "capital_source": capital_source,
        "peak_capital": peak_capital,
        "all_time_high": round(all_time_high, 2),
        "baseline_capital": baseline_capital if baseline_capital > 0 else effective_start,
        "drawdown_pct": round(drawdown_pct, 6),
        "drawdown_limit": drawdown_limit,
        "total_roi": round(total_roi, 6),
    }
    save_state(snapshot)
    return snapshot


def load_portfolio_state(
    *,
    refresh: bool = False,
    log_path: str = TRADES_LOG_PATH,
    starting_balance: Optional[float] = None,
) -> Dict[str, Any]:
    """Load the portfolio state, recomputing it if requested or missing."""
    if refresh:
        return refresh_portfolio_state(log_path=log_path, starting_balance=starting_balance)

    state = load_state()
    if "available_capital" not in state:
        return refresh_portfolio_state(log_path=log_path, starting_balance=starting_balance)
    return state


def get_reinvestment_rate(
    *,
    refresh: bool = False,
    log_path: str = TRADES_LOG_PATH,
    starting_balance: Optional[float] = None,
) -> float:
    """Convenience accessor for the current reinvestment rate."""
    state = load_portfolio_state(
        refresh=refresh,
        log_path=log_path,
        starting_balance=starting_balance,
    )
    return float(state.get("reinvestment_rate", 0.0))


def example_usage() -> None:
    """Print the latest snapshot for quick manual inspection."""
    state = load_portfolio_state(refresh=True)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    example_usage()
