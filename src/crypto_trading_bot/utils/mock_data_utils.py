"""
Mock data utilities (testing only).

This module provides mock market data generators for use in tests and
ad-hoc scripts. It must not be imported by production code paths.
"""

from __future__ import annotations

import random
from datetime import datetime


def generate_mock_data(trading_pair: str) -> dict:
    """
    Generate mock market data for a given trading pair.

    Includes price, RSI, MACD, VWAP, ATR, and Volume. Intended for test
    environments only; do not use in production trading flows.

    Args:
        trading_pair (str): The trading pair (e.g., "BTC-USD")

    Returns:
        dict: A dictionary with simulated market data
    """
    base_prices = {
        "BTC-USD": 30000,
        "ETH-USD": 2000,
        "SOL-USD": 100,
    }

    base_price = base_prices.get(trading_pair, 1000)
    price = base_price * random.uniform(0.95, 1.05)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "pair": trading_pair,
        "price": price,
        "rsi": random.uniform(20, 80),
        "macd": random.uniform(-3, 3),
        "vwap": price * random.uniform(0.98, 1.02),
        "atr": random.uniform(20, 200),
        "volume": random.uniform(1000, 10000),
    }
