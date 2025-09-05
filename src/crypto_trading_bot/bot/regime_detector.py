"""
Market regime detection module.
"""


def detect_regime(prices, window: int = 10):
    """
    Detects market regime based on price history.

    Args:
        prices (list | pd.Series): Historical price series.
        window (int): Rolling window size.

    Returns:
        str: Market regime ('trending', 'volatile', 'choppy', etc.)
    """
    if not prices or len(prices) < window:
        return "unknown"

    # ✅ Ensure prices is a pandas Series
    try:  # Lazy import to avoid hard dependency for linting environments
        import pandas as pd  # pylint: disable=import-error, import-outside-toplevel
    except ImportError:
        return "unknown"
    if isinstance(prices, list):
        prices = pd.Series(prices)

    returns = prices.pct_change().dropna()
    volatility = returns.rolling(window).std().iloc[-1]
    trend = returns.rolling(window).mean().iloc[-1]

    if trend > 0.002 and volatility < 0.01:
        return "uptrend"
    elif trend < -0.002 and volatility < 0.01:
        return "downtrend"
    elif volatility >= 0.01:
        return "volatile"
    else:
        return "choppy"


if __name__ == "__main__":
    test_prices = [100, 101, 102, 104, 107, 106, 108, 110, 111, 113]
    print("Detected regime:", detect_regime(test_prices))
