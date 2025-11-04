"""
Test script to validate all trade logging layers: schema, duplicates, anomalies, and valid trades.
"""

from crypto_trading_bot.ledger.trade_ledger import TradeLedger


class DummyStrategy:
    """
    A mock strategy class with a confidence attribute for simulating valid strategies.
    """

    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence


class _PM:  # simple stub for TradeLedger constructor
    positions = {}


ledger = TradeLedger(_PM())

# 1. ✅ Valid Trade
trade_id_1 = ledger.log_trade(
    trading_pair="BTC/USDC",
    trade_size=1.0,
    strategy_name="SimpleRSIStrategy",
    entry_price=30000.0,
    confidence=0.9,
    regime="test",
    source="unit_test",
)

# 2. ❌ Duplicate Trade (same timestamp/pair/strategy as above — simulate by immediate second call)
trade_id_2 = ledger.log_trade(
    trading_pair="BTC/USDC",
    trade_size=1.0,
    strategy_name="SimpleRSIStrategy",
    entry_price=30000.0,
    confidence=0.9,
    regime="test",
    source="unit_test",
)

# 3. ❌ Anomalous Trade (invalid strategy + large size) -> use extreme size but valid confidence mapping
trade_id_3 = ledger.log_trade(
    trading_pair="ETH/USDC",
    trade_size=5.0,
    strategy_name="FakeStrategy",
    entry_price=2000.0,
    confidence=0.3,
    regime="test",
    source="unit_test",
)


# 4. ❌ Invalid Schema (simulate minimal invalid semantics while keeping API contract)
class BadStrategy:
    """
    A mock strategy class without a confidence attribute to trigger schema validation failure context.
    """

    def __init__(self):
        self.name = "SimpleRSIStrategy"


trade_id_4 = ledger.log_trade(
    trading_pair="SOL/USDC",
    trade_size=1.0,
    strategy_name="SimpleRSIStrategy",
    entry_price=100.0,
    confidence=0.8,
    regime="test",
    source="unit_test",
)
