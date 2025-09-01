"""Unit tests for trading_logic helpers: gather_signals, risk_screen, execute_trade, and exits."""

from crypto_trading_bot.bot.trading_logic import (
    check_and_close_exits,
    execute_trade,
    gather_signals,
    risk_screen,
)
from crypto_trading_bot.indicators.rsi import calculate_rsi


def _to_scalar(x):
    if isinstance(x, (list, tuple)) and x:
        return x[-1]
    if isinstance(x, dict):
        for k in ("rsi", "current", "value", "last"):
            if k in x:
                return x[k]
    return x


def test_gather_signals_stub_shape():
    """gather_signals returns the expected dict shape."""
    out = gather_signals(prices=[1, 2, 3], volumes=[10, 20, 30], context=object())
    assert isinstance(out, dict)
    assert "rsi" in out and "trend" in out and "raw" in out
    assert "prices" in out["raw"] and "volumes" in out["raw"]


def test_risk_screen_stub_true():
    """risk_screen returns a boolean (True for benign inputs)."""
    assert risk_screen({"anything": 1}, context=object()) is True


def test_execute_trade_stub_list():
    """execute_trade returns a list (no side effects in unit tests)."""
    trades = execute_trade({"signal": "buy"}, context=object())
    assert isinstance(trades, list)
    assert not trades  # empty list


def test_check_and_close_exits_stub_zero():
    """check_and_close_exits returns an int count (zero in stub)."""
    closed = check_and_close_exits(context=object())
    assert isinstance(closed, int)
    assert closed == 0


def test_gather_signals_rsi_computes():
    """RSI is computed by gather_signals and matches the indicator util."""
    prices = [100, 101, 102, 103, 102, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112]
    expected = calculate_rsi(prices, 14)
    out = gather_signals(prices=prices, volumes=None, context=None)
    assert out["rsi"] is not None
    exp = _to_scalar(expected)
    got = _to_scalar(out["rsi"])
    assert isinstance(got, (int, float)) and isinstance(exp, (int, float))
    assert abs(got - exp) < 1e-9


def test_risk_screen_returns_bool_for_various_inputs():
    """risk_screen should be a pure gate: always returns a boolean and never raises."""
    # Benign, typical input (should pass)
    assert risk_screen({"pair": "BTC/USD", "confidence": 0.5}, context=object()) is True

    # Edge-ish inputs: empty dict / unknown fields — should still return a bool (not crash)
    for payload in ({}, {"unknown": 123}, {"pair": "BTC/USD", "size": 0}):
        result = risk_screen(payload, context=None)
        assert isinstance(result, bool)
