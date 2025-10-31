"""
market_data_fetcher.py

Provides utility functions for evaluating market conditions and mock data
for testing strategies.
"""

import random


def detect_market_regime(data):
    """
    Determines the current market regime based on RSI, MACD, and volume.

    Args:
        data (dict): Market data with keys: 'rsi', 'macd', 'volume'

    Returns:
        str: One of ['trending', 'choppy', 'low_volatility', 'high_volatility']
    """
    rsi = data.get("rsi")
    macd = data.get("macd")
    volume = data.get("volume")

    if volume is not None:
        if volume < 500:
            return "low_volatility"
        if volume > 5000:
            return "high_volatility"

    if rsi is not None and macd is not None:
        if (rsi > 60 and macd > 0.5) or (rsi < 40 and macd < -0.5):
            return "trending"
        if 40 <= rsi <= 60 and -0.5 < macd < 0.5:
            return "choppy"

    return "unknown"


def get_mock_market_data(_trading_pair: str) -> dict:
    """
    Generate mock market data for testing strategies.

    Args:
        _trading_pair (str): Trading pair (unused, placeholder for future).

    Returns:
        dict: Simulated market data with RSI, MACD, and volume.
    """
    return {
        "rsi": random.randint(30, 70),
        "macd": round(random.uniform(-1, 1), 2),
        "volume": random.randint(100, 10000),
    }


__all__ = ["detect_market_regime", "get_mock_market_data"]
