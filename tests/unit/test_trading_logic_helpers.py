from crypto_trading_bot.bot.trading_logic import (
    check_and_close_exits,
    execute_trade,
    gather_signals,
    risk_screen,
)


def test_gather_signals_stub_shape():
    out = gather_signals(prices=[1, 2, 3], volumes=[10, 20, 30], context=object())
    assert isinstance(out, dict)
    assert "rsi" in out and "trend" in out and "raw" in out
    assert "prices" in out["raw"] and "volumes" in out["raw"]


def test_risk_screen_stub_true():
    assert risk_screen({"anything": 1}, context=object()) is True


def test_execute_trade_stub_list():
    trades = execute_trade({"signal": "buy"}, context=object())
    assert isinstance(trades, list)
    assert trades == []


def test_check_and_close_exits_stub_zero():
    closed = check_and_close_exits(context=object())
    assert isinstance(closed, int)
    assert closed == 0
