"""
Test Script for Trade Ledger Logging

This script simulates a single trade using a dummy strategy object and logs the trade.
Used to verify the output of the `log_trade` function in isolation.
"""

from crypto_trading_bot.ledger.trade_ledger import log_trade


class DummyStrategy:
    """
    Dummy strategy class used to simulate a valid trade with confidence.
    """

    def __init__(self):
        self.confidence = 0.8


log_trade("BTC/USD", 100, DummyStrategy(), "SimpleRSIStrategy", entry_price=29000.0)
