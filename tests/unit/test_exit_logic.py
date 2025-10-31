"""Regression tests for exit logic conditions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crypto_trading_bot.trading_logic import TRADE_INTERVAL, PositionManager


def _position(
    trade_id: str,
    *,
    entry_price: float,
    size: float = 1.0,
    timestamp: datetime | None = None,
    high_water_mark: float | None = None,
) -> dict:
    if timestamp is None:
        ts = datetime.now(timezone.utc).isoformat()
    else:
        ts = timestamp.isoformat()
    position = {
        "trade_id": trade_id,
        "pair": "BTC/USDC",
        "size": size,
        "entry_price": entry_price,
        "timestamp": ts,
        "strategy": "UnitTestStrategy",
        "confidence": 0.8,
        "high_water_mark": high_water_mark or entry_price,
    }
    return position


def test_stop_loss_exit_triggers_below_threshold():
    manager = PositionManager()
    manager.positions["sl-1"] = _position("sl-1", entry_price=100.0)

    exits = manager.check_exits({"BTC/USDC": 98.0}, sl=0.01)

    assert ("sl-1", 98.0, "STOP_LOSS") in exits


def test_take_profit_exit_triggers_above_threshold():
    manager = PositionManager()
    manager.positions["tp-1"] = _position("tp-1", entry_price=100.0)

    exits = manager.check_exits({"BTC/USDC": 102.5}, tp=0.02)

    assert ("tp-1", 102.5, "TAKE_PROFIT") in exits


def test_trailing_stop_respects_high_water_mark():
    manager = PositionManager()
    position = _position("ts-1", entry_price=100.0, high_water_mark=110.0)
    manager.positions["ts-1"] = position

    # Price falls more than trailing stop (1%) from high water mark of 110.
    exits = manager.check_exits({"BTC/USDC": 108.0}, trailing_stop=0.01)

    assert ("ts-1", 108.0, "TRAILING_STOP") in exits


def test_max_hold_exit_after_time_limit():
    manager = PositionManager()
    old_timestamp = datetime.now(timezone.utc) - timedelta(seconds=TRADE_INTERVAL * 20)
    manager.positions["mh-1"] = _position("mh-1", entry_price=100.0, timestamp=old_timestamp)

    exits = manager.check_exits({"BTC/USDC": 100.5}, max_hold_bars=10)

    assert ("mh-1", 100.5, "MAX_HOLD") in exits
