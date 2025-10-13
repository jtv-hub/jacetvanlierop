"""Unit tests for small helpers in the trade ledger module.

Currently validates `_apply_exit_slippage` behavior (no-op + type casting).
"""

from crypto_trading_bot.ledger.trade_ledger import _apply_exit_slippage


def test_apply_exit_slippage_noop_float():
    """Returns the exact same float value unchanged."""
    assert _apply_exit_slippage("BTC/USDC", 12345.67) == 12345.67


def test_apply_exit_slippage_casts_to_float():
    """Casts string inputs to float while preserving numeric value."""
    assert _apply_exit_slippage("ETH/USDC", "100.5") == 100.5
