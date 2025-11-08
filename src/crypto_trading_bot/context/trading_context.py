# src/crypto_trading_bot/context/trading_context.py

"""
Trading Context

Provides the current trading environment, including market regime and reinvestment buffer.
Used to standardize decision-making based on market conditions.
"""

from datetime import datetime, timezone
from typing import Any, Dict

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
        self.regime_snapshot: Dict[str, Any] = {"label": "unknown", "trend_strength": 0.0}
        self.drawdown_pct = 0.0
        self.total_roi = 0.0
        self.available_capital = 0.0
        self.capital_buffer = self.buffer

        self.update_context()

    def update_context(self):
        """
        Refresh regime and buffer information from the persisted portfolio state.
        """
        snapshot = load_portfolio_state(refresh=True)

        regime_label = str(snapshot.get("market_regime", "unknown") or "unknown")
        trend_strength = snapshot.get("trend_strength", 0.0)
        try:
            trend_strength = float(trend_strength)
        except (TypeError, ValueError):
            trend_strength = 0.0
        regime_meta = snapshot.get("regime_meta")
        if not isinstance(regime_meta, dict):
            regime_meta = {}
        regime_meta = dict(regime_meta)
        regime_meta.setdefault("label", regime_label)
        regime_meta.setdefault("trend_strength", trend_strength)
        self.regime_snapshot = regime_meta
        self.regime = regime_label

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
        self.capital_buffer = self.buffer

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

        self.drawdown_pct = float(snapshot.get("drawdown_pct", 0.0) or 0.0)
        self.total_roi = float(snapshot.get("total_roi", 0.0) or 0.0)
        available_capital = snapshot.get("available_capital")
        try:
            self.available_capital = float(available_capital or 0.0)
        except (TypeError, ValueError):
            self.available_capital = 0.0

        self.last_updated = datetime.now(timezone.utc)

    def get_regime(self):
        """Return the current regime snapshot."""
        return self.regime_snapshot

    def get_regime_label(self) -> str:
        """Return the string label for the current regime."""
        if isinstance(self.regime_snapshot, dict):
            return str(self.regime_snapshot.get("label") or self.regime)
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
            "regime": self.get_regime_label(),
            "regime_data": self.regime_snapshot,
            "buffer": self.buffer,
            "buffer_profile": self.buffer_profile,
            "strategy_buffers": self.strategy_buffers,
            "drawdown_pct": self.drawdown_pct,
            "total_roi": self.total_roi,
            "capital_buffer": self.capital_buffer,
            "trend_strength": (
                float(self.regime_snapshot.get("trend_strength", 0.0))
                if isinstance(self.regime_snapshot, dict)
                else 0.0
            ),
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
