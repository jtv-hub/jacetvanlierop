"""
ADX indicator (approximate implementation using close prices).

Note: A precise ADX requires high/low/close; we approximate TR and DM using
close-to-close movements for environments without OHLC data. This is intended
for gating, not precise trading decisions.
"""

from __future__ import annotations

from typing import List


def calculate_adx(prices: List[float], period: int = 14) -> float | None:
    if not prices or len(prices) < period + 2 or any(p is None or p <= 0 for p in prices):
        return None

    # Approximate True Range and directional movements via closes
    trs: List[float] = [0.0]
    plus_dm: List[float] = [0.0]
    minus_dm: List[float] = [0.0]
    for i in range(1, len(prices)):
        up = prices[i] - prices[i - 1]
        down = prices[i - 1] - prices[i]
        trs.append(abs(up))
        plus_dm.append(max(0.0, up))
        minus_dm.append(max(0.0, down))

    def _ema(seq: List[float], p: int) -> List[float]:
        if not seq:
            return []
        k = 2.0 / (p + 1)
        out: List[float] = []
        ema_val = None
        for x in seq:
            ema_val = x if ema_val is None else (x * k + ema_val * (1 - k))
            out.append(ema_val)
        return out

    atr = _ema(trs, period)
    p_dm = _ema(plus_dm, period)
    m_dm = _ema(minus_dm, period)
    if not atr:
        return None
    di_plus: List[float] = []
    di_minus: List[float] = []
    for a, pd, md in zip(atr, p_dm, m_dm):
        if a and a > 0:
            di_plus.append(100.0 * (pd / a))
            di_minus.append(100.0 * (md / a))
        else:
            di_plus.append(0.0)
            di_minus.append(0.0)

    dx: List[float] = []
    for p, m in zip(di_plus, di_minus):
        denom = (p + m) if (p + m) != 0 else 1e-9
        dx.append(100.0 * abs(p - m) / denom)
    adx_series = _ema(dx, period)
    return float(adx_series[-1]) if adx_series else None


__all__ = ["calculate_adx"]
