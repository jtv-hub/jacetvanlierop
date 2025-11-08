"""
state_builder.py
Builds normalized PPO state vectors for both shadow mode and live inference.
Uses indicators, regime info, strategy metadata, portfolio risk + time features.
All features are normalized and versioned to keep PPO policies stable.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

STATE_VERSION = "v2_atr_included"
EMBEDDING_DIM = 8
# 16 base features + embedding slice
STATE_VECTOR_SIZE = 16 + EMBEDDING_DIM

LOGGER = logging.getLogger(__name__)


class StateBuilder:
    """
    Assembles a stable, normalized state vector consumed by the PPO agent.
    Inputs come from:
    - Indicator stack (RSI, EMA diff, MACD, Bollinger width, etc.)
    - Strategy context (strategy_id, confidence score)
    - Regime detector output
    - Portfolio state (drawdown, buffer, win streak)
    - Cyclical time features (hour/day)
    """

    def __init__(self):
        # Placeholders for future StandardScaler / rolling normalization
        self.price_scale = 1.0
        self.feature_scale = 1.0
        self._rng = np.random.default_rng(123)
        self._strategy_embeddings: Dict[str, np.ndarray] = {}

    def _cyclic_time_features(self, dt) -> Dict[str, float]:
        """Return sine/cosine encodings for hour-of-day and day-of-week."""
        hour = dt.hour
        dow = dt.weekday()
        return {
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "dow_sin": math.sin(2 * math.pi * dow / 7),
            "dow_cos": math.cos(2 * math.pi * dow / 7),
        }

    def _strategy_embedding(self, strategy_id: str) -> np.ndarray:
        """Return a deterministic embedding vector per strategy."""
        if strategy_id not in self._strategy_embeddings:
            self._strategy_embeddings[strategy_id] = self._rng.normal(
                0.0,
                0.05,
                EMBEDDING_DIM,
            ).astype(np.float32)
        return self._strategy_embeddings[strategy_id]

    def build(
        self,
        *,
        indicators: Dict[str, float],
        strategy_id: str,
        confidence: float,
        regime: Dict[str, Any],
        portfolio: Dict[str, float],
        timestamp,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          - "vector": np.ndarray
          - "version": STATE_VERSION
          - "components": breakdown used for debugging
        """
        try:
            tfeat = self._cyclic_time_features(timestamp)
            # Strategy ID → deterministic float in [0,1] (temporary until embedding table)
            strategy_float = float(hash(strategy_id) % 9973) / 9973.0
            strategy_embedding = self._strategy_embedding(strategy_id)

            base_vector = np.array(
                [
                    # Indicators
                    indicators.get("rsi", 50.0) / 100.0,
                    indicators.get("ema_diff", 0.0),
                    indicators.get("macd", 0.0),
                    indicators.get("boll_width", 0.0),
                    indicators.get("atr_pct", 0.0),
                    # Strategy + confidence
                    strategy_float,
                    confidence,
                    # Regime
                    regime.get("trend_strength", 0.0),
                    regime.get("volatility_regime", 0.0),
                    # Portfolio
                    portfolio.get("drawdown_pct", 0.0),
                    portfolio.get("capital_buffer", 0.25),
                    portfolio.get("win_streak", 0),
                    # Time
                    tfeat["hour_sin"],
                    tfeat["hour_cos"],
                    tfeat["dow_sin"],
                    tfeat["dow_cos"],
                ],
                dtype=np.float32,
            )

            vector = np.concatenate(
                (base_vector, strategy_embedding),
            ).astype(np.float32, copy=False)

            if vector.size != STATE_VECTOR_SIZE or not np.all(np.isfinite(vector)):
                LOGGER.warning("StateBuilder produced invalid vector; using zeros")
                vector = self._zero_vector()

            return {
                "vector": vector,
                "version": STATE_VERSION,
                "components": {
                    "indicators": indicators,
                    "regime": regime,
                    "portfolio": portfolio,
                    "confidence": confidence,
                    "strategy_id": strategy_id,
                },
            }
        except Exception as exc:
            LOGGER.exception("build_state failure: %s", exc)
            return {
                "vector": self._zero_vector(),
                "version": STATE_VERSION,
                "components": {"error": str(exc)},
            }

    def _zero_vector(self) -> np.ndarray:
        """Return zero vector matching expected PPO input length."""
        return np.zeros(STATE_VECTOR_SIZE, dtype=np.float32)


_STATE_BUILDER = StateBuilder()


def build_state(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build a PPO-ready state from the broader trading context dict."""
    indicators = dict(context.get("indicators") or {})
    regime = dict(context.get("regime") or {})
    portfolio = dict(context.get("portfolio") or {})
    strategy_id = context.get("strategy_id", "unknown")
    confidence = float(context.get("confidence", 0.0))
    timestamp = _coerce_timestamp(context.get("timestamp"))

    return _STATE_BUILDER.build(
        indicators=indicators,
        strategy_id=strategy_id,
        confidence=confidence,
        regime=regime,
        portfolio=portfolio,
        timestamp=timestamp,
    )


def _coerce_timestamp(raw_ts: Any) -> datetime:
    """Convert arbitrary timestamp inputs into timezone-aware UTC datetimes."""
    if isinstance(raw_ts, datetime):
        dt = raw_ts
    elif isinstance(raw_ts, (int, float)):
        try:
            dt = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    elif isinstance(raw_ts, str):
        try:
            dt = datetime.fromisoformat(raw_ts)
        except ValueError:
            dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    else:
        dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect PPO state builder metadata.")
    parser.add_argument("--inspect-vector", action="store_true", help="Print state vector details.")
    cli_args = parser.parse_args()

    if cli_args.inspect_vector:
        info = {
            "state_version": STATE_VERSION,
            "state_vector_size": STATE_VECTOR_SIZE,
            "includes_features": [
                "rsi",
                "ema_diff",
                "macd",
                "boll_width",
                "atr_pct",
                "strategy_confidence",
                "regime_trend_strength",
                "portfolio_drawdown_pct",
                "capital_buffer",
                "time_cyclical_features",
            ],
        }
        print(json.dumps(info, indent=2))
