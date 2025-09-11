"""
Advanced technical strategies: MACD, Keltner Channel Breakout, StochRSI, Bollinger Bands.

Each class implements generate_signal(prices, volume, asset=None) -> dict with keys:
  - signal: 'buy' | 'sell' | None
  - side: same as signal for compatibility
  - confidence: float in [0, 1]
  - strategy: class name

All implementations are dependency-light and operate on close prices only. Some
indicators (ATR, StochRSI) are approximations based on close-to-close ranges
when high/low series are unavailable.
"""

from __future__ import annotations

from math import isfinite
from typing import List, Optional


def _ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series or []
    k = 2.0 / (period + 1)
    out: List[float] = []
    ema_val: Optional[float] = None
    for x in series:
        ema_val = x if ema_val is None else (x * k + ema_val * (1 - k))
        out.append(ema_val)
    return out


def _sma(series: List[float], window: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for x in series:
        q.append(x)
        s += x
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _stdev(series: List[float], window: int) -> List[float]:
    out: List[float] = []
    buf: List[float] = []
    for x in series:
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        n = len(buf)
        if n <= 1:
            out.append(0.0)
        else:
            mean = sum(buf) / n
            var = sum((v - mean) ** 2 for v in buf) / (n - 1)
            out.append(var**0.5)
    return out


def _atr_from_close(series: List[float], period: int = 10) -> List[float]:
    # Approximate ATR using absolute close-to-close changes
    trs = [0.0]
    for i in range(1, len(series)):
        trs.append(abs(series[i] - series[i - 1]))
    return _ema(trs, period)


def _rsi_series(prices: List[float], period: int = 14) -> List[float]:
    gains: List[float] = [0.0]
    losses: List[float] = [0.0]
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(0.0, diff))
        losses.append(max(0.0, -diff))
    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)
    rsi: List[float] = []
    for g, loss in zip(avg_gain, avg_loss):
        if loss == 0:
            rsi.append(100.0)
        else:
            rs = g / loss
            rsi.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi


class MACDStrategy:
    """MACD crossover strategy.

    Parameters
    - fast: EMA period for fast line
    - slow: EMA period for slow line
    - signal: EMA period for signal line (applied to MACD line)
    - per_asset: optional overrides keyed by asset symbol
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        per_asset: dict | None = None,
    ):
        self.fast = fast
        self.slow = slow
        self.signal_p = signal
        self.per_asset = per_asset or {}

    def generate_signal(
        self,
        prices: List[float],
        volume,
        asset: str | None = None,
    ):
        """Generate a trading signal using MACD crossovers.

        Arguments
        - prices: list of close prices (most-recent last)
        - volume: kept for interface compatibility (unused)
        - asset: optional asset symbol for per-asset overrides

        Returns a dict with keys: signal, side, confidence, strategy.
        """
        # Accept volume for interface compatibility but do not use it here
        del volume

        if not prices or len(prices) < max(self.slow, self.signal_p) + 2 or any(p is None or p <= 0 for p in prices):
            return {
                "signal": None,
                "side": None,
                "confidence": 0.0,
                "strategy": "MACDStrategy",
            }
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            self.fast = int(cfg.get("fast", self.fast))
            self.slow = int(cfg.get("slow", self.slow))
            self.signal_p = int(cfg.get("signal", self.signal_p))

        ema_fast = _ema(prices, self.fast)
        ema_slow = _ema(prices, self.slow)
        macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
        signal_line = _ema(macd_line, self.signal_p)
        if len(macd_line) < 2 or len(signal_line) < 2:
            return {
                "signal": None,
                "side": None,
                "confidence": 0.0,
                "strategy": "MACDStrategy",
            }
        m0, m1 = macd_line[-2], macd_line[-1]
        s0, s1 = signal_line[-2], signal_line[-1]
        diff = m1 - s1
        # Confidence scales with normalized distance between MACD and signal
        conf = min(1.0, max(0.0, abs(diff) / (abs(prices[-1]) * 0.001)))
        if m0 <= s0 and m1 > s1:
            return {
                "signal": "buy",
                "side": "buy",
                "confidence": float(conf),
                "strategy": "MACDStrategy",
            }
        if m0 >= s0 and m1 < s1:
            return {
                "signal": "sell",
                "side": "sell",
                "confidence": float(conf),
                "strategy": "MACDStrategy",
            }
        return {
            "signal": None,
            "side": None,
            "confidence": 0.0,
            "strategy": "MACDStrategy",
        }


class KeltnerBreakoutStrategy:
    """Keltner Channel breakout strategy using EMA basis and ATR width.

    Parameters
    - ema_window: EMA window for channel basis
    - atr_period: ATR period (approximated from close-to-close)
    - atr_mult: multiplier for channel width
    - per_asset: optional overrides keyed by asset symbol
    """

    def __init__(
        self,
        ema_window: int = 20,
        atr_period: int = 10,
        atr_mult: float = 2.0,
        per_asset: dict | None = None,
    ):
        self.ema_window = ema_window
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.per_asset = per_asset or {}

    def generate_signal(
        self,
        prices: List[float],
        volume,
        asset: str | None = None,
    ):
        """Generate breakout signals based on Keltner Channels.

        Arguments
        - prices: list of close prices (most-recent last)
        - volume: kept for interface compatibility (unused)
        - asset: optional asset symbol for per-asset overrides

        Returns a dict with keys: signal, side, confidence, strategy.
        """
        # Accept volume for interface compatibility but do not use it here
        del volume

        if (
            not prices
            or len(prices) < max(self.ema_window, self.atr_period) + 2
            or any(p is None or p <= 0 for p in prices)
        ):
            return {
                "signal": None,
                "side": None,
                "confidence": 0.0,
                "strategy": "KeltnerBreakoutStrategy",
            }
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            self.ema_window = int(cfg.get("ema_window", self.ema_window))
            self.atr_period = int(cfg.get("atr_period", self.atr_period))
            self.atr_mult = float(cfg.get("atr_mult", self.atr_mult))

        basis = _ema(prices, self.ema_window)[-1]
        atr = _atr_from_close(prices, self.atr_period)[-1]
        upper = basis + self.atr_mult * atr
        lower = basis - self.atr_mult * atr
        px = prices[-1]
        width = max(1e-9, upper - lower)
        if px > upper:
            conf = min(1.0, (px - upper) / width)
            return {
                "signal": "buy",
                "side": "buy",
                "confidence": float(conf),
                "strategy": "KeltnerBreakoutStrategy",
            }
        if px < lower:
            conf = min(1.0, (lower - px) / width)
            return {
                "signal": "sell",
                "side": "sell",
                "confidence": float(conf),
                "strategy": "KeltnerBreakoutStrategy",
            }
        return {
            "signal": None,
            "side": None,
            "confidence": 0.0,
            "strategy": "KeltnerBreakoutStrategy",
        }


class StochRSIStrategy:
    """Stochastic RSI strategy using RSI and a stochastic normalization window.

    Parameters
    - rsi_period: lookback for RSI
    - stoch_period: window for stochastic normalization of RSI
    - per_asset: optional overrides keyed by asset symbol
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        per_asset: dict | None = None,
    ):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.per_asset = per_asset or {}

    def generate_signal(
        self,
        prices: List[float],
        volume,
        asset: str | None = None,
    ):
        """Generate signals based on Stochastic RSI levels and slope.

        Arguments
        - prices: list of close prices (most-recent last)
        - volume: kept for interface compatibility (unused)
        - asset: optional asset symbol for per-asset overrides

        Returns a dict with keys: signal, side, confidence, strategy.
        """
        # Accept volume for interface compatibility but do not use it here
        del volume

        if (
            not prices
            or len(prices) < max(self.rsi_period + self.stoch_period + 2, 10)
            or any(p is None or p <= 0 for p in prices)
        ):
            return {
                "signal": None,
                "side": None,
                "confidence": 0.0,
                "strategy": "StochRSIStrategy",
            }
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            self.rsi_period = int(cfg.get("rsi_period", self.rsi_period))
            self.stoch_period = int(cfg.get("stoch_period", self.stoch_period))

        rsi_seq = _rsi_series(prices, self.rsi_period)
        rsi_seq = rsi_seq[-(self.stoch_period + 2) :]
        if len(rsi_seq) < self.stoch_period + 2:
            return {
                "signal": None,
                "side": None,
                "confidence": 0.0,
                "strategy": "StochRSIStrategy",
            }
        recent = rsi_seq[-self.stoch_period :]
        rsi_now = rsi_seq[-1]
        lo = min(recent)
        hi = max(recent)
        denom = max(1e-9, (hi - lo))
        stoch = (rsi_now - lo) / denom
        # Rising if last value > previous
        rising = rsi_seq[-1] > rsi_seq[-2]
        falling = rsi_seq[-1] < rsi_seq[-2]
        if stoch < 0.2 and rising:
            conf = min(1.0, (0.2 - stoch) / 0.2)
            return {
                "signal": "buy",
                "side": "buy",
                "confidence": float(conf),
                "strategy": "StochRSIStrategy",
            }
        if stoch > 0.8 and falling:
            conf = min(1.0, (stoch - 0.8) / 0.2)
            return {
                "signal": "sell",
                "side": "sell",
                "confidence": float(conf),
                "strategy": "StochRSIStrategy",
            }
        return {
            "signal": None,
            "side": None,
            "confidence": 0.0,
            "strategy": "StochRSIStrategy",
        }


class BollingerBandStrategy:
    """Bollinger Bands strategy with light RSI alignment.

    Parameters
    - window: SMA/stdev window
    - mult: standard deviation multiplier
    - per_asset: optional overrides keyed by asset symbol
    """

    def __init__(
        self,
        window: int = 20,
        mult: float = 2.0,
        per_asset: dict | None = None,
    ):
        self.window = window
        self.mult = mult
        self.per_asset = per_asset or {}

    def generate_signal(
        self,
        prices: List[float],
        volume,
        asset: str | None = None,
    ):
        """Generate signals based on Bollinger Band touches with RSI filter.

        Arguments
        - prices: list of close prices (most-recent last)
        - volume: kept for interface compatibility (unused)
        - asset: optional asset symbol for per-asset overrides

        Returns a dict with keys: signal, side, confidence, strategy.
        """
        # Accept volume for interface compatibility but do not use it here
        del volume

        if not prices or len(prices) < self.window + 2 or any(p is None or p <= 0 for p in prices):
            return {
                "signal": None,
                "side": None,
                "confidence": 0.0,
                "strategy": "BollingerBandStrategy",
            }
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            self.window = int(cfg.get("window", self.window))
            self.mult = float(cfg.get("mult", self.mult))

        ma = _sma(prices, self.window)[-1]
        sd = _stdev(prices, self.window)[-1]
        upper = ma + self.mult * sd
        lower = ma - self.mult * sd
        px = prices[-1]
        width = max(1e-9, upper - lower)
        # Light RSI alignment using our internal RSI sequence
        rsi_now = _rsi_series(prices, 14)[-1]
        if px < lower and rsi_now > 30:
            conf = min(1.0, (lower - px) / width)
            return {
                "signal": "buy",
                "side": "buy",
                "confidence": float(conf),
                "strategy": "BollingerBandStrategy",
            }
        if px > upper and rsi_now < 70:
            conf = min(1.0, (px - upper) / width)
            return {
                "signal": "sell",
                "side": "sell",
                "confidence": float(conf),
                "strategy": "BollingerBandStrategy",
            }
        return {
            "signal": None,
            "side": None,
            "confidence": 0.0,
            "strategy": "BollingerBandStrategy",
        }


__all__ = [
    "MACDStrategy",
    "KeltnerBreakoutStrategy",
    "StochRSIStrategy",
    "BollingerBandStrategy",
]


class VWAPStrategy:
    """Simple mean-reversion to a rolling VWAP using closes and synthetic volumes.

    Falls back to SMA if volume is not provided.
    """

    def __init__(self, window: int = 20, per_asset: dict | None = None):
        self.window = window
        self.per_asset = per_asset or {}

    def generate_signal(self, prices: List[float], volume, asset: str | None = None, **_):
        del _
        if not prices or len(prices) < max(self.window, 5) or any(p is None or p <= 0 for p in prices):
            return {"signal": None, "side": None, "confidence": 0.0, "strategy": "VWAPStrategy"}
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            self.window = int(cfg.get("window", self.window))
        # Approximate VWAP using available volume or uniform weights
        w = []
        n = len(prices[-self.window :])
        if volume is None:
            w = [1.0] * n
        else:
            try:
                w = [max(1.0, float(volume))] * n
            except Exception:
                w = [1.0] * n
        px = prices[-self.window :]
        vwap = sum(p * w[i] for i, p in enumerate(px)) / sum(w)
        last = prices[-1]
        band = max(1e-9, vwap * 0.004)
        conf = min(1.0, abs(last - vwap) / (band * 2))
        print(f"[STRATEGY DEBUG] VWAP vwap={vwap:.6f} last={last:.6f} conf={conf:.3f}")
        if last < vwap - band:
            return {"signal": "buy", "side": "buy", "confidence": float(conf), "strategy": "VWAPStrategy"}
        if last > vwap + band:
            return {"signal": "sell", "side": "sell", "confidence": float(conf), "strategy": "VWAPStrategy"}
        return {"signal": None, "side": None, "confidence": 0.0, "strategy": "VWAPStrategy"}


class ADXStrategy:
    """Trend-strength breakout using ADX value.

    Uses ADX to gate: if strong trend and recent momentum confirms, trigger.
    Expects `adx` passed in generate_signal kwargs; otherwise returns hold.
    """

    def __init__(self, threshold: float = 25.0, per_asset: dict | None = None):
        self.threshold = threshold
        self.per_asset = per_asset or {}

    def generate_signal(self, prices: List[float], volume, asset: str | None = None, adx: float | None = None, **_):
        del _, volume
        if not prices or any(p is None or p <= 0 for p in prices):
            return {"signal": None, "side": None, "confidence": 0.0, "strategy": "ADXStrategy"}
        if asset and asset in self.per_asset:
            cfg = self.per_asset[asset]
            self.threshold = float(cfg.get("threshold", self.threshold))
        if adx is None or not isfinite(adx):
            print("[STRATEGY DEBUG] ADXStrategy missing adx; holding")
            return {"signal": None, "side": None, "confidence": 0.0, "strategy": "ADXStrategy"}
        last = prices[-1]
        prev = prices[-2] if len(prices) >= 2 else last
        momentum = (last - prev) / max(prev, 1e-9)
        conf = min(1.0, max(0.0, (abs(momentum) * max(0.0, adx - self.threshold)) * 10))
        print(f"[STRATEGY DEBUG] ADXStrategy adx={adx:.2f} mom={momentum:.5f} conf={conf:.3f}")
        if adx >= max(self.threshold, 25.0):
            if momentum > 0:
                return {"signal": "buy", "side": "buy", "confidence": float(conf), "strategy": "ADXStrategy"}
            if momentum < 0:
                return {"signal": "sell", "side": "sell", "confidence": float(conf), "strategy": "ADXStrategy"}
        return {"signal": None, "side": None, "confidence": 0.0, "strategy": "ADXStrategy"}


class CompositeStrategy:
    """Weighted composite of RSI (proxied), MACD, and ADX inputs.

    This lightweight version reuses MACD and ADXStrategy signals and merges confidences.
    """

    def __init__(self, per_asset: dict | None = None):
        self.per_asset = per_asset or {}
        self.macd = MACDStrategy()
        self.adxs = ADXStrategy()

    def generate_signal(self, prices: List[float], volume, asset: str | None = None, adx: float | None = None, **_):
        del _
        m = self.macd.generate_signal(prices, volume=volume, asset=asset)
        a = self.adxs.generate_signal(prices, volume=volume, asset=asset, adx=adx)
        # Merge signals: prefer when they agree; otherwise hold with low confidence.
        signal = None
        conf = 0.0
        if m.get("signal") and a.get("signal") and m["signal"] == a["signal"]:
            signal = m["signal"]
            conf = min(1.0, (m.get("confidence", 0.0) + a.get("confidence", 0.0)) / 2.0)
        elif m.get("signal"):
            signal = m["signal"]
            conf = float(m.get("confidence", 0.0)) * 0.5
        elif a.get("signal"):
            signal = a["signal"]
            conf = float(a.get("confidence", 0.0)) * 0.5
        print(f"[STRATEGY DEBUG] Composite m={m.get('signal')} a={a.get('signal')} conf={conf:.3f}")
        return {
            "signal": signal,
            "side": signal,
            "confidence": float(conf),
            "strategy": "CompositeStrategy",
        }
