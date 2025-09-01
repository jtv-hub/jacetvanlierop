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


def test_risk_screen_min_checks():
    """risk_screen blocks/permits based on RSI, position cap, and cash buffer."""

    class Portfolio:
        def __init__(self, opens, cash, equity):
            self.open_positions = [object()] * opens
            self.cash = cash
            self.equity = equity

    class Ctx:
        def __init__(self, portfolio=None, cfg=None):
            self.portfolio = portfolio
            self.config = cfg or {"risk": {"max_open_positions": 3, "capital_buffer": 0.25}}

    # RSI guard (blocks high RSI buys)
    from crypto_trading_bot.bot.trading_logic import risk_screen

    assert risk_screen({"signal": "buy", "rsi": 75}, context=Ctx()) is False
    assert risk_screen({"signal": "buy", "rsi": 65}, context=Ctx()) is True

    # Position cap guard
    assert risk_screen({"signal": "buy"}, context=Ctx(Portfolio(3, 1000, 4000))) is False
    assert risk_screen({"signal": "buy"}, context=Ctx(Portfolio(2, 1000, 4000))) is True

    # Cash buffer guard
    # 12.5% < 25%  -> block
    assert risk_screen({"signal": "buy"}, context=Ctx(Portfolio(0, 500, 4000))) is False
    # 40%  >= 25%  -> allow
    assert risk_screen({"signal": "buy"}, context=Ctx(Portfolio(0, 600, 1500))) is True
