"""
simple_rsi_strategies.py

Implements the SimpleRSIStrategy class using RSI thresholds.
"""

from typing import List

from crypto_trading_bot.indicators.rsi import calculate_rsi


class SimpleRSIStrategy:
    """
    RSI-based strategy with dynamic confidence scoring.

    Confidence is computed based on distance from the RSI band defined by
    (oversold=lower, overbought=upper). Fallback to 0.5 only if RSI is missing.
    """

    def __init__(
        self,
        period: int = 21,
        lower: float = 30.0,
        upper: float = 70.0,
        per_asset: dict | None = None,
    ):
        """Initialize with optional per-asset parameter overrides.

        per_asset example:
            {"ETH": {"lower": 32, "upper": 68, "period": 21}}
        """
        self.period = period
        self.lower = lower  # oversold threshold
        self.upper = upper  # overbought threshold
        self.per_asset = per_asset or {}

    def validate_volume(self, volume, min_volume: int = 100) -> bool:
        """
        Validates that the provided volume meets a minimum threshold.

        Args:
            volume (int or float): The current volume to validate.
            min_volume (int): The minimum acceptable volume level. Defaults to 500.

        Returns:
            bool: True if the volume is sufficient, False otherwise.
        """
        if volume is None or volume < min_volume:
            print(f"[DEBUG] Low volume: {volume} < {min_volume}")
            return False
        return True

    def generate_signal(self, prices: List[float], volume, asset: str | None = None):
        """
        Generates a buy, sell, or hold signal based on RSI and volume filters.

        Args:
            prices (list or np.array): Historical price data.
            volume (float or int): Latest volume value.

        Returns:
            dict: A dictionary containing the signal type and confidence level.
        """
        # Filter 1: Price validity and sufficiency
        if not prices or len(prices) < max(self.period + 1, 5) or any(p is None or p <= 0 for p in prices):
            print(f"[DEBUG] SimpleRSIStrategy invalid prices: len={len(prices) if prices else 0}")
            return {"signal": None, "confidence": 0.0}

        # Filter 2: Volume check
        if not self.validate_volume(volume):
            return {"signal": None, "confidence": 0.0}

        # Apply per-asset params if provided
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            self.period = int(cfg.get("period", self.period))
            self.lower = float(cfg.get("lower", self.lower))
            self.upper = float(cfg.get("upper", self.upper))

        rsi = calculate_rsi(prices, self.period)
        print(f"[DEBUG] RSI: {rsi}, Recent Prices: {prices[-5:]}")
        print(f"[DEBUG] Thresholds: oversold={self.lower}, overbought={self.upper}")

        # Fallback if RSI missing/invalid
        try:
            rsi_val = float(rsi) if rsi is not None else None
        except (TypeError, ValueError):
            rsi_val = None

        if rsi_val is None:
            return {"signal": None, "confidence": 0.5}

        # Nonlinear confidence centered at RSI midpoint (50)
        # Tunable curve: logistic with center at d0 (distance=20 -> ~0.5),
        # and steepness k. We zero confidence inside a neutral band [45,55].
        oversold = float(min(self.lower, self.upper))
        overbought = float(max(self.lower, self.upper))
        d = abs(rsi_val - 50.0)
        if d <= 5.0:
            conf_nl = 0.0
        else:
            import math

            d0 = 20.0  # distance to midpoint where conf ~= 0.5
            k = 5.0  # steepness of curve; lower -> steeper
            s = 1.0 / (1.0 + math.exp(-(d - d0) / k))
            s_neutral = 1.0 / (1.0 + math.exp(-((5.0) - d0) / k))
            # Rescale so s_neutral -> 0, and 1 remains 1
            conf_nl = (s - s_neutral) / max(1e-9, (1.0 - s_neutral))
            conf_nl = max(0.0, min(1.0, conf_nl))

        # Signals: still trigger at thresholds, but confidence uses nonlinear curve
        if rsi_val < oversold:
            return {
                "signal": "buy",
                "side": "buy",
                "strategy": "SimpleRSIStrategy",
                "confidence": float(conf_nl if conf_nl > 0 else 0.0),
            }
        if rsi_val > overbought:
            return {
                "signal": "sell",
                "side": "sell",
                "strategy": "SimpleRSIStrategy",
                "confidence": float(conf_nl if conf_nl > 0 else 0.0),
            }
        return {"signal": None, "side": None, "strategy": "SimpleRSIStrategy", "confidence": 0.0}
