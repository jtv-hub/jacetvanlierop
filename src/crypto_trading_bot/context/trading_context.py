# src/crypto_trading_bot/context/trading_context.py

"""
Trading Context

Provides the current trading environment, including market regime and reinvestment buffer.
Used to standardize decision-making based on market conditions.
"""

from datetime import datetime, timezone
from typing import Dict

from crypto_trading_bot.bot.state.portfolio_state import load_portfolio_state
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
        self.buffer_profile: Dict[str, float] = {}
        self.strategy_buffers: Dict[str, Dict[str, float]] = {}
        self._adx_cache: dict[str, float] = {}

        self.update_context()

    def update_context(self):
        """
        Refresh regime and buffer information from the persisted portfolio state.
        """
        snapshot = load_portfolio_state(refresh=True)

        regime = snapshot.get("market_regime", "unknown")
        self.regime = str(regime) if isinstance(regime, str) else "unknown"

        raw_profile = snapshot.get("regime_capital_buffers") or {}
        self.buffer_profile = {}
        for key, value in raw_profile.items():
            if isinstance(value, (int, float)):
                self.buffer_profile[key] = float(value)

        defaults = CONFIG.get("buffer_defaults", {})
        fallback_buffer = float(defaults.get(self.regime, defaults.get("unknown", 0.25)))
        capital_buffer = snapshot.get("capital_buffer")
        if isinstance(capital_buffer, (int, float)):
            self.buffer = float(capital_buffer)
        elif self.buffer_profile:
            self.buffer = float(self.buffer_profile.get(self.regime, fallback_buffer))
        else:
            self.buffer = fallback_buffer

        raw_strategy_buffers = snapshot.get("strategy_buffers") or {}
        parsed: Dict[str, Dict[str, float]] = {}
        for strategy, data in raw_strategy_buffers.items():
            if not isinstance(data, dict):
                continue
            parsed[strategy] = {}
            for regime_key, value in data.items():
                if isinstance(value, (int, float)):
                    parsed[strategy][regime_key] = float(value)
        self.strategy_buffers = parsed

        self.last_updated = datetime.now(timezone.utc)

    def get_regime(self):
        """Returns the current market regime."""
        return self.regime

    def get_buffer(self):
        """Returns the current reinvestment buffer based on regime."""
        return self.buffer

    def get_buffer_for_strategy(self, strategy_name: str | None = None) -> float:
        """Return a regime-aware buffer, honoring strategy-specific overrides."""
        if not strategy_name:
            return self.buffer
        if strategy_name in self.strategy_buffers:
            strategy_profile = self.strategy_buffers[strategy_name]
            return strategy_profile.get(self.regime, self.buffer)
        if self.buffer_profile:
            return self.buffer_profile.get(self.regime, self.buffer)
        return self.buffer

    def get_snapshot(self):
        """Returns a snapshot dictionary of the current context (timestamp, regime, buffer)."""
        return {
            "timestamp": self.last_updated.isoformat(),
            "regime": self.regime,
            "buffer": self.buffer,
            "buffer_profile": self.buffer_profile,
            "strategy_buffers": self.strategy_buffers,
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
