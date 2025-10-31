"""
rsi.py

Implements the Relative Strength Index (RSI) calculation for trading strategies.
"""

import json
from datetime import datetime, timezone

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency guard
    np = None  # type: ignore[assignment]

from crypto_trading_bot.utils.log_rotation import get_anomalies_logger

anomalies_logger = get_anomalies_logger()


def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Args:
        prices (list or np.array): Historical price data.
        period (int, optional): Lookback period for RSI calculation. Defaults to 14.

    Returns:
        float: RSI value between 0 and 100

    Raises:
        ValueError: If input is invalid or RSI cannot be computed safely.
    """
    # Input validation: ensure list-like, sufficient length, and positive prices
    if np is None:  # pragma: no cover - requires numpy installed
        raise ImportError("numpy is required for RSI calculations")

    if prices is None:
        raise ValueError("RSI: prices is None")
    if period is None or period <= 0:
        raise ValueError(f"RSI: invalid period {period}")
    if not hasattr(prices, "__len__") or len(prices) < period + 1:
        got = len(prices) if hasattr(prices, "__len__") else 0
        print(f"[RSI ERROR] Not enough candles for RSI: got {got}, need {period + 1}")
        msg = f"RSI: insufficient data len={len(prices) if hasattr(prices, '__len__') else 'n/a'}"
        raise ValueError(msg + f" < {period+1}")
    if any(p is None or p <= 0 for p in prices):
        raise ValueError("RSI: non-positive or None price encountered")

    prices = np.asarray(prices, dtype=float)

    # Differences between consecutive prices
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Average gain/loss over first 'period'
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # Avoid divide-by-zero using Wilder's standard behavior; do not early-return
    eps = 1e-12
    # Debug samples for gains/losses and seed averages
    try:
        print(
            "RSI DEBUG seed:",
            float(avg_gain),
            float(avg_loss),
            "gains_sample=",
            np.asarray(gains[: min(5, len(gains))]).tolist(),
            "losses_sample=",
            np.asarray(losses[: min(5, len(losses))]).tolist(),
        )
    except (ValueError, TypeError):
        # Debug printing should not interfere with computation
        pass
    if avg_gain < eps and avg_loss < eps:
        rs = 1.0  # flat -> RSI ~ 50
    elif avg_loss < eps:
        rs = float("inf")  # no losses -> RSI 100
    elif avg_gain < eps:
        rs = 0.0  # no gains -> RSI 0
    else:
        rs = avg_gain / avg_loss

    # Initialize RSI list with first value after seed period
    rsi_values = []
    # Compute first RSI based on seed averages
    if avg_gain < eps and avg_loss < eps:
        first_rsi = 50.0
    elif avg_loss < eps:
        first_rsi = 100.0
    elif avg_gain < eps:
        first_rsi = 0.0
    elif not np.isfinite(rs) or rs < 0:
        # unexpected numeric state; log and fallback
        try:
            anomalies_logger.info(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": "RSI Invalid RS",
                        "avg_gain": float(avg_gain),
                        "avg_loss": float(avg_loss),
                        "rs": str(rs),
                    },
                    separators=(",", ":"),
                )
            )
        except (ValueError, TypeError):
            # Best-effort anomaly logging
            pass
        first_rsi = 50.0
    else:
        first_rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi_values.append(first_rsi)

    # Wilder's smoothing for subsequent values
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_gain < eps and avg_loss < eps:
            rsi = 50.0
            rs_dbg = 1.0
        elif avg_loss < eps:
            rsi = 100.0
            rs_dbg = float("inf")
        elif avg_gain < eps:
            rsi = 0.0
            rs_dbg = 0.0
        else:
            rs_dbg = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs_dbg))
        # Debug log before clamping
        # (debug) print suppressed
        rsi_values.append(rsi)

    # Final RSI before clamping
    rsi_out = float(rsi_values[-1]) if rsi_values else 50.0

    # Clamp after validations and log invalid values
    if not np.isfinite(rsi_out):
        try:
            anomalies_logger.info(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": "RSI Invalid Output",
                        "value": str(rsi_out),
                    },
                    separators=(",", ":"),
                )
            )
        except (ValueError, TypeError):
            # Best-effort anomaly logging
            pass
        rsi_out = 50.0

    rsi_out = max(0.0, min(100.0, rsi_out))

    # Defensive: surface if RSI ended up None for any reason
    if rsi_out is None:  # pragma: no cover - defensive guard
        print("[RSI ERROR] RSI calculation returned None — possible malformed prices")
        return None

    return rsi_out
