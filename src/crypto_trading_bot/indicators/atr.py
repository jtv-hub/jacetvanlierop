"""Average True Range helper utilities."""

from __future__ import annotations

from typing import Iterable, List


def compute_atr(prices: Iterable[float], period: int = 14) -> float:
    """Return naive ATR using absolute close deltas over ``period`` samples."""

    price_list: List[float] = [float(p) for p in prices if p is not None]
    if not price_list or len(price_list) <= max(1, period):
        return 0.0
    tr_values = [abs(price_list[i] - price_list[i - 1]) for i in range(1, len(price_list))]
    if not tr_values:
        return 0.0
    window = tr_values[-period:]
    return sum(window) / len(window)
