"""
state_builder.py

Constructs a normalized 60-dimensional state vector using multi-timeframe
price action features. For each of the 1m, 5m, and 15m intervals we pull the
latest 20 OHLCV candles, derive close, volume, and RSI series (3 features × 20
steps = 60 per timeframe), concatenate them (180 dim), then downsample to a
compact 60-length vector.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List

import numpy as np

from crypto_trading_bot.indicators.rsi import calculate_rsi
from crypto_trading_bot.utils.kraken_api import get_ohlc_data

LOGGER = logging.getLogger(__name__)

TIMEFRAMES: tuple[int, ...] = (1, 5, 15)
CANDLES_PER_FRAME = 20
FEATURES_PER_CANDLE = 3  # close, volume, RSI
STATE_DIM = 60
FULL_DIM = len(TIMEFRAMES) * CANDLES_PER_FRAME * FEATURES_PER_CANDLE  # 180
USE_LIVE_DATA = os.getenv("STATE_BUILDER_USE_LIVE", "0").strip().lower() in {"1", "true", "yes"}


def _normalize(series: np.ndarray) -> np.ndarray:
    """Return mean-centered / std-scaled copy clipped to [-5, 5]."""

    arr = np.asarray(series, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        arr = arr - mean
    else:
        arr = (arr - mean) / std
    return np.clip(arr, -5.0, 5.0)


def _rsi_series(closes: Iterable[float], period: int = 14) -> np.ndarray:
    """Compute RSI for each candle using rolling window with last value carry."""

    prices = np.asarray(list(closes), dtype=np.float64)
    if prices.size < period + 1:
        raise ValueError(f"Need at least {period+1} closes, got {prices.size}")

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi_values: List[float] = []

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    eps = 1e-12

    def _rs_to_rsi(gain: float, loss: float) -> float:
        if gain < eps and loss < eps:
            return 50.0
        if loss < eps:
            return 100.0
        if gain < eps:
            return 0.0
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    rsi_values.append(_rs_to_rsi(avg_gain, avg_loss))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi_values.append(_rs_to_rsi(avg_gain, avg_loss))

    # Front-fill the initial slots (for i < period) with first RSI
    first_value = rsi_values[0] if rsi_values else 50.0
    prefix = np.full(period, first_value, dtype=np.float32)
    tail = np.asarray(rsi_values, dtype=np.float32)
    out = np.concatenate([prefix, tail], dtype=np.float32)
    if out.size > prices.size:
        out = out[-prices.size :]
    if out.size < prices.size:
        out = np.pad(out, (prices.size - out.size, 0), constant_values=first_value)
    return np.clip(out[-prices.size :], 0.0, 100.0)


def _mock_candles(pair: str, interval: int) -> list[dict[str, float]]:
    """Deterministic synthetic candles to keep unit tests offline-friendly."""

    seed = abs(hash((pair.upper(), interval))) % (2**31)
    rng = np.random.default_rng(seed)
    base_price = 20000.0 + (seed % 5000)
    base_volume = 1500.0 + (seed % 200)

    prices = base_price * np.exp(rng.standard_normal(CANDLES_PER_FRAME) * 0.002).cumprod()
    volumes = base_volume * (1 + rng.standard_normal(CANDLES_PER_FRAME) * 0.1)
    volumes = np.clip(volumes, 10.0, None)

    candles: list[dict[str, float]] = []
    ts_base = 1_700_000_000
    for idx in range(CANDLES_PER_FRAME):
        candles.append(
            {
                "time": ts_base + idx * interval * 60,
                "open": float(prices[idx - 1] if idx > 0 else prices[idx]),
                "high": float(prices[idx] * (1 + rng.random() * 0.001)),
                "low": float(prices[idx] * (1 - rng.random() * 0.001)),
                "close": float(prices[idx]),
                "volume": float(volumes[idx]),
            }
        )
    return candles


def _load_timeframe(pair: str, interval: int) -> list[dict[str, float]]:
    if USE_LIVE_DATA:
        try:
            candles = get_ohlc_data(pair, interval=interval, limit=CANDLES_PER_FRAME)
            if len(candles) < CANDLES_PER_FRAME:
                raise ValueError("Not enough candles returned")
            return candles[-CANDLES_PER_FRAME:]
        except Exception as exc:  # pragma: no cover - network/IO path
            LOGGER.warning(
                "Falling back to mock candles for %s interval=%s (%s)",
                pair,
                interval,
                exc,
            )
    return _mock_candles(pair, interval)


def _build_features(pair: str, interval: int) -> np.ndarray:
    candles = _load_timeframe(pair, interval)
    closes = np.array([c.get("close", 0.0) for c in candles], dtype=np.float32)
    volumes = np.array([c.get("volume", 0.0) for c in candles], dtype=np.float32)
    closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
    volumes = np.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        rsi_values = _rsi_series(closes, period=min(14, max(2, len(closes) - 1)))
    except Exception:
        try:
            rsi_single = calculate_rsi(closes.tolist(), period=14)
            rsi_values = np.full_like(closes, float(rsi_single), dtype=np.float32)
        except Exception:
            rsi_values = np.full_like(closes, 50.0, dtype=np.float32)

    close_norm = _normalize(closes)
    vol_norm = _normalize(volumes)
    rsi_norm = _normalize(rsi_values)

    return np.concatenate([close_norm, vol_norm, rsi_norm]).astype(np.float32)


def _reduce(full: np.ndarray) -> np.ndarray:
    """Reduce 180-d vector to 60 by averaging each consecutive triple."""

    reshaped = full.reshape(60, 3)
    return reshaped.mean(axis=1).astype(np.float32)


def build_state_vector(pair: str) -> np.ndarray:
    """Return a 60-dimensional normalized state vector for ``pair``."""

    if not isinstance(pair, str) or "/" not in pair:
        raise ValueError("pair must be in format 'BASE/QUOTE'")

    feature_blocks = [_build_features(pair.upper(), interval) for interval in TIMEFRAMES]
    full_vector = np.concatenate(feature_blocks)
    if full_vector.size != FULL_DIM:
        raise RuntimeError(f"Expected {FULL_DIM} features, got {full_vector.size}")
    reduced = _reduce(full_vector)
    if reduced.shape != (STATE_DIM,):
        raise RuntimeError(f"State vector shape mismatch: {reduced.shape}")
    return reduced


__all__ = ["build_state_vector", "STATE_DIM"]
