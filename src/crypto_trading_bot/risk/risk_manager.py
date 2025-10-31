"""Dynamic risk buffer calculations based on realized performance."""

from __future__ import annotations

from math import prod
from typing import Iterable, List, Tuple

try:
    from crypto_trading_bot.portfolio_state import load_closed_trades
except ImportError:  # pragma: no cover - defensive fallback
    load_closed_trades = None  # type: ignore[assignment]

TRADES_LOG_PATH = "logs/trades.log"


def _get_closed_trade_rois(max_trades: int = 250) -> List[Tuple[float, float]]:
    """Return a list of (roi, confidence) tuples for recent closed trades."""
    if load_closed_trades is None:
        return []

    trades = load_closed_trades(TRADES_LOG_PATH)
    trades.sort(key=lambda t: t.get("timestamp") or "")
    recent = trades[-max_trades:]
    rois: List[Tuple[float, float]] = []
    for trade in recent:
        try:
            roi = float(trade.get("roi"))
            confidence = float(trade.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        rois.append((roi, confidence))
    return rois


def _compute_drawdown(rois: Iterable[float]) -> float:
    """Return absolute maximum drawdown from a sequence of ROI values."""
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for roi in rois:
        equity *= 1.0 + roi
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown = (equity - peak) / peak
            if drawdown < max_dd:
                max_dd = drawdown
    return abs(max_dd)


def get_dynamic_buffer() -> float:
    """Derive a capital buffer multiplier using closed-trade performance metrics."""
    samples = _get_closed_trade_rois()
    if not samples:
        return 0.35

    rois = [roi for roi, _ in samples]
    confidences = [conf for _, conf in samples if conf is not None]

    trade_count = len(rois)
    wins = sum(1 for roi in rois if roi > 0)
    win_rate = wins / trade_count if trade_count else 0.0
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    drawdown = _compute_drawdown(rois)
    cumulative_return = prod(1.0 + roi for roi in rois) - 1.0

    buffer = 1.0

    if trade_count < 10:
        buffer = 0.4
    elif drawdown >= 0.35:
        buffer = 0.2
    elif drawdown >= 0.25:
        buffer = 0.3
    elif drawdown >= 0.15:
        buffer = 0.5
    elif drawdown >= 0.08:
        buffer = 0.7

    if win_rate < 0.4:
        buffer = min(buffer, 0.25)
    elif win_rate < 0.5:
        buffer = min(buffer, 0.4)

    if cumulative_return < -0.05:
        buffer = min(buffer, 0.3)
    elif cumulative_return > 0.25 and drawdown < 0.1 and win_rate > 0.55:
        buffer = max(buffer, 0.85)

    if average_confidence < 0.35:
        buffer = min(buffer, 0.5)

    return max(0.15, min(buffer, 1.0))


__all__ = ["get_dynamic_buffer"]
