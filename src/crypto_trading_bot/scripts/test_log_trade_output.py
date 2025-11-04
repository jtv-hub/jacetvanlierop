"""
Test Script for Trade Ledger Logging

This script simulates a single trade using a dummy strategy object and logs the trade.
Updated to use the class-based TradeLedger API.
"""

from crypto_trading_bot.ledger.trade_ledger import TradeLedger


class DummyStrategy:
    """
    Dummy strategy class used to simulate a valid trade with confidence.
    """

    def __init__(self):
        self.confidence = 0.8


class _PM:
    positions = {}


ledger = TradeLedger(_PM())

trade_id = ledger.log_trade(
    trading_pair="BTC/USDC",
    trade_size=1.0,
    strategy_name="SimpleRSIStrategy",
    entry_price=29000.0,
    confidence=0.8,
    regime="test",
    source="unit_test",
)
