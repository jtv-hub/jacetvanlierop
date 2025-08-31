"""
Test script for validating the trade ledger logging functionality.
Runs a few sample trades and prints their logged output.
"""

from crypto_trading_bot.ledger.trade_ledger import log_trade


def main():
    """
    Execute sample trades for BTC, ETH, and SOL to test the ledger logging.
    """
    # Run a few forced trades with different ROI outcomes
    print("🚀 Running trade ledger test...")

    # Trade 1 - BTC
    trade1 = log_trade(
        trading_pair="BTC-USD",
        trade_size=100,
        strategy_name="TestStrategy",
        confidence=0.95,
        price=40000,
        volume=1500,
        indicators={"RSI": 55},
        regime="uptrend",
    )
    print(trade1)

    # Trade 2 - ETH
    trade2 = log_trade(
        trading_pair="ETH-USD",
        trade_size=200,
        strategy_name="TestStrategy",
        confidence=0.88,
        price=2500,
        volume=1200,
        indicators={"MACD": "bullish"},
        regime="choppy",
    )
    print(trade2)

    # Trade 3 - SOL
    trade3 = log_trade(
        trading_pair="SOL-USD",
        trade_size=150,
        strategy_name="TestStrategy",
        confidence=0.90,
        price=100,
        volume=500,
        indicators={"ATR": 2.5},
        regime="unknown",
    )
    print(trade3)


if __name__ == "__main__":
    main()
