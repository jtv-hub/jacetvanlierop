import json
from pathlib import Path

import pytest


def test_trading_logic_exits_only_mode_initializes_executed_trades():
    mod = pytest.importorskip("crypto_trading_bot.trading_logic")
    try:
        mod.evaluate_signals_and_trade(check_exits_only=True)
    except UnboundLocalError as e:
        pytest.fail(f"executed_trades should be initialized: {e}")


def test_trading_logic_position_manager_check_exits_interface():
    mod = pytest.importorskip("crypto_trading_bot.trading_logic")
    pm = mod.PositionManager()
    trade_id = "T-1"
    pm.positions[trade_id] = {
        "trade_id": trade_id,
        "pair": "BTC/USDC",
        "size": 0.001,
        "entry_price": 20000.0,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "strategy": "SimpleRSIStrategy",
        "confidence": 0.65,
        "high_water_mark": 20000.0,
    }
    exits = pm.check_exits({"BTC/USDC": 19980.0}, tp=0.02, sl=0.001, trailing_stop=0.05, max_hold_bars=0)
    assert isinstance(exits, list)


def test_trade_ledger_log_trade_validation_confidence_range():
    ledger_mod = pytest.importorskip("crypto_trading_bot.ledger.trade_ledger")

    class DummyPM:
        positions = {}

    tl = ledger_mod.TradeLedger(DummyPM())
    with pytest.raises(ValueError):
        tl.log_trade(trading_pair="BTC/USDC", trade_size=0.001, strategy_name="S", confidence=1.5)


def test_trade_ledger_update_trade_idempotent(tmp_path, monkeypatch):
    ledger_mod = pytest.importorskip("crypto_trading_bot.ledger.trade_ledger")

    class DummyPM:
        positions = {}

    tl = ledger_mod.TradeLedger(DummyPM())
    tid = tl.log_trade(
        trading_pair="BTC/USDC",
        trade_size=0.001,
        strategy_name="TestStrategy",
        confidence=0.7,
        entry_price=20000.0,
    )
    tl.update_trade(trade_id=tid, exit_price=20500.0, reason="TAKE_PROFIT")
    tl.update_trade(trade_id=tid, exit_price=20500.0, reason="TAKE_PROFIT")  # idempotent
    trade = tl.trade_index.get(tid)
    assert trade is not None
    assert trade.get("reason") == "tp_hit"
    assert trade.get("reason_display") == "TAKE_PROFIT"


def test_trade_ledger_detects_duplicate(monkeypatch):
    ledger_mod = pytest.importorskip("crypto_trading_bot.ledger.trade_ledger")

    class DummyPM:
        positions = {}

    log_path = Path("logs/trades.log")
    pos_path = Path("logs/positions.jsonl")
    log_path.unlink(missing_ok=True)
    pos_path.unlink(missing_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    seed_trade = {
        "trade_id": "test_001",
        "pair": "BTC/USDC",
        "side": "buy",
        "entry_price": 100.0,
        "exit_price": 105.0,
        "size": 0.01,
        "capital_used": 1.0,
        "roi": 0.05,
        "exit_reason": "tp",
        "strategy": "unit_test",
        "confidence": 0.5,
        "status": "closed",
        "timestamp": "2024-01-01T00:00:00+00:00",
    }
    log_path.write_text(json.dumps(seed_trade) + "\n", encoding="utf-8")

    tl = ledger_mod.TradeLedger(DummyPM())
    tl.reload_trades()

    original_window = ledger_mod._DUPLICATE_WINDOW_SECONDS
    ledger_mod._DUPLICATE_WINDOW_SECONDS = 300
    try:
        duplicate_id = tl.log_trade(
            trading_pair="BTC/USDC",
            trade_size=0.01,
            strategy_name="unit_test",
            confidence=0.5,
            entry_price=100.0,
            side="buy",
            trade_id=seed_trade["trade_id"],
        )
    finally:
        ledger_mod._DUPLICATE_WINDOW_SECONDS = original_window

    assert duplicate_id == seed_trade["trade_id"]
    tl.trades.clear()
    tl.trade_index.clear()
    tl.txid_index.clear()
    tl.reload_trades()
    assert len(tl.trades) == 1


def test_portfolio_risk_correlation_threshold_precedence():
    pr_mod = pytest.importorskip("crypto_trading_bot.utils.portfolio_risk")
    signals = [
        {"asset": "BTC", "signal_score": 0.9, "regime": "trending", "strategy": "S"},
        {"asset": "ETH", "signal_score": 0.85, "regime": "trending", "strategy": "S"},
    ]
    prices = {"BTC": 20000.0, "ETH": 1000.0}
    trades = pr_mod.portfolio_risk_logic(100_000.0, signals, prices, max_exposure=0.06, max_trades=2)
    assert isinstance(trades, list)


def test_learning_machine_import_and_smoke():
    _ = pytest.importorskip("crypto_trading_bot.learning.learning_machine")


def test_rsi_invalid_input_raises():
    import pytest

    from crypto_trading_bot.indicators.rsi import calculate_rsi

    with pytest.raises(ValueError):
        calculate_rsi([], period=14)
    with pytest.raises(ValueError):
        calculate_rsi([100.0] * 10, period=14)


def test_rsi_valid_input_range():
    from crypto_trading_bot.indicators.rsi import calculate_rsi

    prices = [100 + (i % 3 - 1) for i in range(50)]
    rsi = calculate_rsi(prices, period=14)
    assert 0.0 <= rsi <= 100.0
