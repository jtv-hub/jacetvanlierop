"""
market_data.py
Generates mock or real market data for the trading bot.
Initially supports mock data generation for testing purposes.
"""

import random
from datetime import datetime


def generate_mock_data(trading_pair: str) -> dict:
    """
    Generate mock market data for a given trading pair.
    Includes price, RSI, MACD, VWAP, ATR, and Volume.

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
        "rsi": random.uniform(20, 80),  # Overbought/oversold oscillator
        "macd": random.uniform(-3, 3),  # Trend momentum
        "vwap": price * random.uniform(0.98, 1.02),  # VWAP around price
        "atr": random.uniform(20, 200),  # Volatility measure
        "volume": random.uniform(1000, 10000),  # Random trading volume
    }


if __name__ == "__main__":
    # Quick test for BTC, ETH, and SOL
    for pair in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        print(f"📊 Mock data for {pair}:")
        print(generate_mock_data(pair))
        print("-" * 40)
