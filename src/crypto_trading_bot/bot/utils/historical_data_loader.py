"""
historical_data_loader.py

Utility module for loading mock historical price data for use in backtesting
trading strategies in the crypto trading bot framework.
"""

import random


def _wobble_series(
    start: float, steps: int, step_min: float, step_max: float
) -> list[float]:
    vals = [start]
    for _ in range(1, steps):
        vals.append(
            vals[-1] + random.choice([-1, 1]) * random.uniform(step_min, step_max)
        )
    return vals


def load_dummy_price_data():
    """
    Loads mock historical price data for backtesting purposes.
    Returns:
        dict: { asset_symbol (str): list of prices (float) }
    """
    return {
        "BTC": _wobble_series(30000.0, 200, 20.0, 120.0),
        "ETH": _wobble_series(1800.0, 200, 2.0, 15.0),
        "SOL": _wobble_series(20.0, 200, 0.05, 0.5),
    }
