# generate_mock_data.py
"""
Generate mock OHLCV data for testing Trade Replay.
Creates CSVs for BTC, ETH, and SOL in /data folder.
"""

import os
from datetime import datetime, timedelta

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def generate_mock_data(pair_name: str, days: int = 1):
    """Generate random OHLCV data for a given trading pair."""
    # Local imports to avoid hard dependency at module import time
    # Local imports avoid hard dependency at module import time
    import numpy as np  # pylint: disable=import-error, import-outside-toplevel
    import pandas as pd  # pylint: disable=import-error, import-outside-toplevel

    end = datetime.now()
    start = end - timedelta(days=days)
    timestamps = pd.date_range(start=start, end=end, freq="T")  # 1-minute candles

    # Generate synthetic prices
    prices = np.cumsum(np.random.randn(len(timestamps))) + 100
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices + np.random.rand(len(prices)),
            "high": prices + np.random.rand(len(prices)) * 2,
            "low": prices - np.random.rand(len(prices)) * 2,
            "close": prices,
            "volume": np.random.randint(1, 1000, len(prices)),
        }
    )

    filename = pair_name.replace("/", "-") + ".csv"
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"✅ Generated mock data: {path}")


if __name__ == "__main__":
    for pair in ["BTC/USDC", "ETH/USDC", "SOL/USDC"]:
        generate_mock_data(pair, days=1)
