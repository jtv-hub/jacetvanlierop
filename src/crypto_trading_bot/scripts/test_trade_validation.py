"""
Test script to validate all trade logging layers: schema, duplicates, anomalies, and valid trades.
"""

from crypto_trading_bot.ledger.trade_ledger import log_trade


class DummyStrategy:
    """
    A mock strategy class with a confidence attribute for simulating valid strategies.
    """

    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence


# 1. ✅ Valid Trade
log_trade("BTC/USDC", 100, DummyStrategy("SimpleRSIStrategy", 0.9), "SimpleRSIStrategy")

# 2. ❌ Duplicate Trade (same timestamp/pair/strategy as above — simulate by overriding timestamp)
log_trade("BTC/USDC", 100, DummyStrategy("SimpleRSIStrategy", 0.9), "SimpleRSIStrategy")

# 3. ❌ Anomalous Trade (invalid strategy + large size)
log_trade("ETH/USDC", 9999, DummyStrategy("FakeStrategy", 0.5), "FakeStrategy")


# 4. ❌ Invalid Schema (simulate by using an invalid strategy instance that lacks confidence)
class BadStrategy:
    """
    A mock strategy class without a confidence attribute to trigger schema validation failure.
    """

    def __init__(self):
        self.name = "SimpleRSIStrategy"


log_trade("SOL/USDC", 100, BadStrategy(), "SimpleRSIStrategy")
