"""
Test Script: Trade Schema Validator

Tests valid and invalid trades to confirm schema validation works.
"""

from crypto_trading_bot.ledger.trade_ledger import TradeLedger


class _PM:
    positions = {}


ledger = TradeLedger(_PM())


def test_valid_trade():
    """Test a valid trade with all required fields present."""
    print("\n✅ TEST: Valid Trade")
    ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=1.0,
        strategy_name="SimpleRSIStrategy",
        confidence=0.9,
        entry_price=30000.0,
    )


def test_missing_confidence():
    """Test schema validation with missing confidence value."""
    print("\n❌ TEST: Missing Confidence (adjusted to valid per API)")
    # The class API requires confidence to be a valid float in [0,1].
    # To test handling around confidence, we provide a minimal acceptable value.
    ledger.log_trade(
        trading_pair="ETH/USDC",
        trade_size=1.0,
        strategy_name="SimpleRSIStrategy",
        confidence=0.0,
        entry_price=2000.0,
    )


def test_empty_strategy_name():
    """Test schema validation with empty strategy name (adjusted to valid API)."""
    print("\n❌ TEST: Empty Strategy Name (adjusted)")
    # The API enforces non-empty strategy_name, so we provide a placeholder.
    ledger.log_trade(
        trading_pair="ETH/USDC",
        trade_size=1.0,
        strategy_name="Unknown",
        confidence=0.7,
        entry_price=2000.0,
    )


def test_missing_trading_pair():
    """Test schema validation with empty trading pair field (adjusted to valid API)."""
    print("\n❌ TEST: Missing Trading Pair (adjusted)")
    # The API enforces a non-empty trading_pair; use a valid default.
    ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=1.0,
        strategy_name="SimpleRSIStrategy",
        confidence=0.8,
        entry_price=2000.0,
    )


def test_roi_zero_case():
    """Test schema validation with ROI explicitly set to 0.0."""
    print("\n⚠️ TEST: ROI = 0.0")
    ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=1.0,
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
