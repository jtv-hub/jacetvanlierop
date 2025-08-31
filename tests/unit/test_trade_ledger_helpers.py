from crypto_trading_bot.ledger.trade_ledger import _apply_exit_slippage


def test_apply_exit_slippage_noop_float():
    assert _apply_exit_slippage("BTC/USD", 12345.67) == 12345.67


def test_apply_exit_slippage_casts_to_float():
    assert _apply_exit_slippage("ETH/USD", "100.5") == 100.5
