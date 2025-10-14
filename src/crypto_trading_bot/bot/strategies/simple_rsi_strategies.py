"""
simple_rsi_strategies.py

Implements the SimpleRSIStrategy class using RSI thresholds.
"""

import math
from typing import List

from crypto_trading_bot.indicators.rsi import calculate_rsi
from crypto_trading_bot.utils.system_logger import get_system_logger

logger = get_system_logger().getChild("strategies.simple_rsi")


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
            logger.debug(
                "Volume below minimum threshold",
                extra={"volume": volume, "min_volume": min_volume},
            )
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
        base_period = self.period
        base_lower = self.lower
        base_upper = self.upper

        # Filter 1: Price validity and sufficiency
        if not prices:
            logger.debug("RSI evaluation skipped: no price history available")
            return {"signal": None, "confidence": 0.0}
        if len(prices) < max(base_period + 1, 5):
            logger.debug(
                "RSI evaluation skipped: insufficient price history",
                extra={"history_length": len(prices)},
            )
            return {"signal": None, "confidence": 0.0}
        if any(p is None or p <= 0 for p in prices):
            logger.debug("RSI evaluation skipped: invalid price detected")
            return {"signal": None, "confidence": 0.0}

        # Filter 2: Volume check
        if not self.validate_volume(volume):
            return {"signal": None, "confidence": 0.0}

        period = base_period
        lower = base_lower
        upper = base_upper

        # Apply per-asset params if provided
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            try:
                period = int(cfg.get("period", period))
            except (TypeError, ValueError):
                period = base_period
            try:
                lower = float(cfg.get("lower", lower))
            except (TypeError, ValueError):
                lower = base_lower
            try:
                upper = float(cfg.get("upper", upper))
            except (TypeError, ValueError):
                upper = base_upper

        period = max(2, period)

        rsi = calculate_rsi(prices, period)

        # Fallback if RSI missing/invalid
        try:
            rsi_val = float(rsi) if rsi is not None else None
        except (TypeError, ValueError):
            rsi_val = None
        logger.info(
            "RSI evaluation",
            extra={
                "asset": asset,
                "price": prices[-1],
                "volume": volume,
                "oversold": lower,
                "overbought": upper,
                "period": period,
                "rsi": rsi_val,
            },
        )

        if rsi_val is None:
            return {"signal": None, "confidence": 0.0}

        # Nonlinear confidence centered at RSI midpoint (50)
        # Tunable curve: logistic with center at d0 (distance=20 -> ~0.5),
        # and steepness k. We zero confidence inside a neutral band [45,55].
        oversold = float(min(lower, upper))
        overbought = float(max(lower, upper))
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
                logger.info(
                    "ADX below threshold; skipping trade",
                    extra={"asset": asset_label, "adx": adx, "rsi": rsi_val},
                )
                return {
                    "signal": None,
                    "side": None,
                    "strategy": "SimpleRSIStrategy",
                    "confidence": 0.0,
                }
            # Strong trend: boost confidence
            if adx > 40:
                baseline_conf = min(1.0, baseline_conf * 1.2)
                logger.info(
                    "ADX strong trend adjustment applied",
                    extra={
                        "asset": asset_label,
                        "adx": adx,
                        "confidence": baseline_conf,
                        "rsi": rsi_val,
                    },
                )
            else:
                logger.info(
                    "ADX neutral adjustment",
                    extra={
                        "asset": asset_label,
                        "adx": adx,
                        "confidence": baseline_conf,
                        "rsi": rsi_val,
                    },
                )

        # Signals: still trigger at thresholds, but confidence uses nonlinear curve
        def _scaled_conf(score: float) -> float:
            return max(0.4, min(0.9, 0.4 + score * 0.5))

        final_conf = _scaled_conf(baseline_conf)

        if rsi_val < oversold and baseline_conf > 0.0:
            logger.info(
                "RSI buy signal triggered",
                extra={
                    "asset": asset,
                    "rsi": rsi_val,
                    "confidence": final_conf,
                    "oversold": oversold,
                    "period": period,
                },
            )
            return {
                "signal": "buy",
                "side": "buy",
                "strategy": "SimpleRSIStrategy",
                "confidence": final_conf,
                "raw_confidence": round(baseline_conf, 4),
            }
        if rsi_val > overbought and baseline_conf > 0.0:
            logger.info(
                "RSI sell signal triggered",
                extra={
                    "asset": asset,
                    "rsi": rsi_val,
                    "confidence": final_conf,
                    "overbought": overbought,
                    "period": period,
                },
            )
            return {
                "signal": "sell",
                "side": "sell",
                "strategy": "SimpleRSIStrategy",
                "confidence": final_conf,
                "raw_confidence": round(baseline_conf, 4),
            }
        logger.info(
            "RSI hold condition",
            extra={
                "asset": asset,
                "rsi": rsi_val,
                "confidence": baseline_conf,
                "oversold": oversold,
                "overbought": overbought,
            },
        )
        return {"signal": None, "side": None, "strategy": "SimpleRSIStrategy", "confidence": 0.0}
