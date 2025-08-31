"""
simple_rsi_strategies.py

Implements the SimpleRSIStrategy class using RSI thresholds.
"""

from typing import List

from crypto_trading_bot.indicators.rsi import calculate_rsi


class SimpleRSIStrategy:
    """
    A simple RSI-based strategy that generates buy, sell, or hold signals
    based on relative strength index thresholds.
    """

    def __init__(self, period=21, lower=48, upper=75):
        self.period = period
        self.lower = lower
        self.upper = upper

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

    def generate_signal(self, prices: List[float], volume):
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

        rsi = calculate_rsi(prices, self.period)
        # Debug line per instructions
        print(f"[DEBUG] RSI: {rsi}, Recent Prices: {prices[-5:]}")
        print(f"[DEBUG] Thresholds: lower={self.lower}, upper={self.upper}")
        if rsi is None:
            return {"signal": None, "confidence": 0.0}

        signal = None
        confidence = 0.0

        # Shared confidence scale: 0.4 (min) .. 1.0 (max)
        def scale(min_v: float, max_v: float, ratio: float) -> float:
            r = max(0.0, min(1.0, ratio))
            return min_v + (max_v - min_v) * r

        if rsi < self.lower:
            signal = "buy"
            # Distance below lower threshold, normalized to [0,1]
            ratio = (self.lower - rsi) / max(self.lower, 1.0)
            confidence = scale(0.4, 1.0, ratio)
        elif rsi > self.upper:
            signal = "sell"
            ratio = (rsi - self.upper) / max(100.0 - self.upper, 1.0)
            confidence = scale(0.4, 1.0, ratio)
        else:
            signal = None  # neutral/hold region
            confidence = 0.0

        return {"signal": signal, "confidence": float(confidence)}
