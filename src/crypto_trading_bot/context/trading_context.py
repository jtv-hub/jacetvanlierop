# src/crypto_trading_bot/context/trading_context.py

"""
Trading Context

Provides the current trading environment, including market regime and reinvestment buffer.
Used to standardize decision-making based on market conditions.
"""

import random
from datetime import datetime, timezone

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.technical_indicators.adx import calculate_adx


class TradingContext:
    """
    Tracks current market regime and reinvestment buffer.
    Supports dynamic strategy adjustment.
    """

    def __init__(self):
        self.last_updated = datetime.now(timezone.utc)
        self.regime = "unknown"
        self.buffer = 0.25  # Default buffer
        self._adx_cache: dict[str, float] = {}

        self.update_context()

    def update_context(self):
        """
        Stub for regime detection logic.
        Replace with actual logic in future versions (e.g. volatility scans, trend detection).
        """
        possible_regimes = ["trending", "chop", "volatile", "flat", "unknown"]
        self.regime = random.choice(possible_regimes)

        # Assign buffer based on regime via CONFIG
        defaults = CONFIG.get("buffer_defaults", {})
        self.buffer = float(defaults.get(self.regime, defaults.get("unknown", 0.25)))

        self.last_updated = datetime.now(timezone.utc)

    def get_regime(self):
        """Returns the current market regime."""
        return self.regime

    def get_buffer(self):
        """Returns the current reinvestment buffer based on regime."""
        return self.buffer

    def get_snapshot(self):
        """Returns a snapshot dictionary of the current context (timestamp, regime, buffer)."""
        return {
            "timestamp": self.last_updated.isoformat(),
            "regime": self.regime,
            "buffer": self.buffer,
        }

    def get_adx(
        self,
        pair: str,
        prices: list[float] | None = None,
        period: int = 14,
    ) -> float | None:
        """Compute or return cached ADX for a pair using recent closes.

        The caller can pass preloaded prices to avoid re-fetching.
        """
        try:
            if pair in self._adx_cache:
                return self._adx_cache.get(pair)
            if not prices:
                return None
            val = calculate_adx(prices, period=period)
            if val is not None:
                self._adx_cache[pair] = float(val)
            return val
        except ValueError as e:
            print(f"[ERROR] Failed to compute ADX for {pair}: {e}")
            return None
