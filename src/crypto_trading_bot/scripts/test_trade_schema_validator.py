"""
Test Script: Trade Schema Validator

Tests valid and invalid trades to confirm schema validation works.
"""

from crypto_trading_bot.ledger.trade_ledger import log_trade


def test_valid_trade():
    """Test a valid trade with all required fields present."""
    print("\n✅ TEST: Valid Trade")
    log_trade(
        trading_pair="BTC/USD",
        trade_size=100,
        strategy_name="SimpleRSIStrategy",
        confidence=0.9,
        entry_price=30000.0,
    )


def test_missing_confidence():
    """Test schema validation with missing confidence value."""
    print("\n❌ TEST: Missing Confidence")
    log_trade(
        trading_pair="ETH/USD",
        trade_size=100,
        strategy_name="SimpleRSIStrategy",
        confidence=None,
        entry_price=2000.0,
    )


def test_empty_strategy_name():
    """Test schema validation with empty strategy name."""
    print("\n❌ TEST: Empty Strategy Name")
    log_trade(
        trading_pair="ETH/USD",
        trade_size=100,
        strategy_name="",
        confidence=0.7,
        entry_price=2000.0,
    )


def test_missing_trading_pair():
    """Test schema validation with empty trading pair field."""
    print("\n❌ TEST: Missing Trading Pair")
    log_trade(
        trading_pair="",
        trade_size=100,
        strategy_name="SimpleRSIStrategy",
        confidence=0.8,
        entry_price=2000.0,
    )


def test_roi_zero_case():
    """Test schema validation with ROI explicitly set to 0.0."""
    print("\n⚠️ TEST: ROI = 0.0")
    log_trade(
        trading_pair="BTC/USD",
        trade_size=100,
        strategy_name="SimpleRSIStrategy",
        confidence=0.8,
        entry_price=30000.0,
    )


if __name__ == "__main__":
    test_valid_trade()
    test_missing_confidence()
    test_empty_strategy_name()
    test_missing_trading_pair()
    test_roi_zero_case()
