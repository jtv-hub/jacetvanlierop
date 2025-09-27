"""
simple_rsi_strategies.py

Implements the SimpleRSIStrategy class using RSI thresholds.
"""

import logging
import math
from typing import List

from crypto_trading_bot.indicators.rsi import calculate_rsi

logger = logging.getLogger("SimpleRSIStrategy")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.FileHandler("logs/full_debug.log")
    formatter = logging.Formatter(r"\[%(levelname)s\] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SimpleRSIStrategy:
    """
    RSI-based strategy with dynamic confidence scoring.

    Confidence is computed based on distance from the RSI band defined by
    (oversold=lower, overbought=upper). Missing RSI data yields a zero-confidence hold.
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

    def generate_signal(
        self,
        prices: List[float],
        volume,
        asset: str | None = None,
        adx: float | None = None,
    ):
        """
        Generates a buy, sell, or hold signal based on RSI and volume filters.

        Args:
            prices (list or np.array): Historical price data.
            volume (float or int): Latest volume value.

        Returns:
            dict: A dictionary containing the signal type and confidence level.
        """
        # Filter 1: Price validity and sufficiency
        if not prices:
            logger.debug("[RSI DEBUG] Exit: prices is empty")
            return {"signal": None, "confidence": 0.0}
        if len(prices) < max(self.period + 1, 5):
            logger.debug("[RSI DEBUG] Exit: not enough prices (len=%s)", len(prices))
            return {"signal": None, "confidence": 0.0}
        if any(p is None or p <= 0 for p in prices):
            logger.debug("[RSI DEBUG] Exit: invalid price found")
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
        # Explicit RSI debug prints per asset
        if rsi is not None:
            print(f"[RSI DEBUG] {asset}: RSI={rsi}")
        else:
            print(f"[RSI DEBUG] {asset}: RSI is None")
        # Format RSI value safely without broad exception
        try:
            rsi_num = float(rsi) if rsi is not None else None
        except (TypeError, ValueError):
            rsi_num = None
        rsi_str = f"{rsi_num:.2f}" if rsi_num is not None else str(rsi)
        logger.debug(
            "[RSI DEBUG] %s RSI=%s price=%s volume=%s",
            asset,
            rsi_str,
            prices[-1],
            volume,
        )
        print(f"[DEBUG] Thresholds: oversold={self.lower}, overbought={self.upper}")

        # Fallback if RSI missing/invalid
        try:
            rsi_val = float(rsi) if rsi is not None else None
        except (TypeError, ValueError):
            rsi_val = None

        if rsi_val is None:
            return {"signal": None, "confidence": 0.0}

        # Nonlinear confidence centered at RSI midpoint (50)
        # Tunable curve: logistic with center at d0 (distance=20 -> ~0.5),
        # and steepness k. We zero confidence inside a neutral band [45,55].
        oversold = float(min(self.lower, self.upper))
        overbought = float(max(self.lower, self.upper))
        d = abs(rsi_val - 50.0)
        if d <= 5.0:
            conf_nl = 0.0
        else:
            d0 = 20.0  # distance to midpoint where confidence transitions meaningfully
            k = 5.0  # steepness of curve; lower -> steeper
            s = 1.0 / (1.0 + math.exp(-(d - d0) / k))
            s_neutral = 1.0 / (1.0 + math.exp(-((5.0) - d0) / k))
            # Rescale so s_neutral -> 0, and 1 remains 1
            conf_nl = (s - s_neutral) / max(1e-9, (1.0 - s_neutral))
            conf_nl = max(0.0, min(1.0, conf_nl))

        # Apply ADX regime filter after RSI computed, before decisions
        baseline_conf = float(conf_nl if conf_nl > 0 else 0.0)

        if adx is not None:
            asset_label = asset or ""
            # Weak trend: skip trades
            if adx < 20:
                print(f"[SKIP] {asset_label}: ADX too weak ({adx})")
                print(f"[ADX DEBUG] {asset_label}: ADX={adx}, adjusted confidence=0.0")
                return {
                    "signal": None,
                    "side": None,
                    "strategy": "SimpleRSIStrategy",
                    "confidence": 0.0,
                }
            # Strong trend: boost confidence
            if adx > 40:
                baseline_conf = min(1.0, baseline_conf * 1.2)
                print(f"[ADX DEBUG] {asset_label}: ADX={adx}, adjusted confidence={baseline_conf}")
            else:
                print(f"[ADX DEBUG] {asset_label}: ADX={adx}, adjusted confidence={baseline_conf}")

        # Signals: still trigger at thresholds, but confidence uses nonlinear curve
        if rsi_val < oversold and baseline_conf > 0.0:
            return {
                "signal": "buy",
                "side": "buy",
                "strategy": "SimpleRSIStrategy",
                "confidence": 1.0,
                "raw_confidence": round(baseline_conf, 4),
            }
        if rsi_val > overbought and baseline_conf > 0.0:
            return {
                "signal": "sell",
                "side": "sell",
                "strategy": "SimpleRSIStrategy",
                "confidence": 1.0,
                "raw_confidence": round(baseline_conf, 4),
            }
        return {"signal": None, "side": None, "strategy": "SimpleRSIStrategy", "confidence": 0.0}
