"""
reinvestment.py

Handles dynamic reinvestment logic based on market conditions and
portfolio value milestones. Designed to help the trading bot adjust
capital allocation from profits intelligently.
"""


def calculate_reinvestment_rate(portfolio_value: float, market_regime: str) -> float:
    """
    Determines reinvestment rate based on market regime and portfolio value.

    Args:
        portfolio_value (float): Current value of the portfolio in USD.
        market_regime (str): One of ["trending", "volatile", "flat"].

    Returns:
        float: Reinvestment rate as a decimal (e.g., 0.5 for 50%).
    """
    if portfolio_value < 250000:
        if market_regime == "trending":
            return 1.0  # 100% reinvestment
        elif market_regime == "volatile":
            return 0.5  # 50%
        elif market_regime == "flat":
            return 0.25  # 25%
    else:
        if market_regime == "trending":
            return 0.5
        else:
            return 0.25


def example_usage():
    """
    Demonstrates the reinvestment rate output for different market regimes
    and portfolio values.
    """
    regimes = ["trending", "volatile", "flat"]
    for regime in regimes:
        rate_100k = calculate_reinvestment_rate(100000, regime) * 100
        rate_300k = calculate_reinvestment_rate(300000, regime) * 100
        print(f"Portfolio = $100K, Regime = {regime} → Reinvest {rate_100k:.0f}%")
        print(f"Portfolio = $300K, Regime = {regime} → Reinvest {rate_300k:.0f}%")


if __name__ == "__main__":
    example_usage()
