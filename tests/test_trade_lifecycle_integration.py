"""Integration test covering trade open → exit → ledger logging → learning update."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from crypto_trading_bot.bot.trading_logic import PositionManager
from crypto_trading_bot.learning import learning_machine
from crypto_trading_bot.ledger import trade_ledger as trade_ledger_module
from crypto_trading_bot.ledger.trade_ledger import TradeLedger


def test_trade_lifecycle_integration(tmp_path, monkeypatch, request):
    """End-to-end verification that a trade flows into learning metrics."""

    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    log_path = log_dir / "trades.log"
    positions_log_path = log_dir / "positions.jsonl"

    original_trades_log_path = trade_ledger_module.TRADES_LOG_PATH
    original_positions_path = trade_ledger_module.POSITIONS_PATH
    original_handlers = list(trade_ledger_module.trade_logger.handlers)

    monkeypatch.setattr(trade_ledger_module, "TRADES_LOG_PATH", str(log_path))
    monkeypatch.setattr(trade_ledger_module, "POSITIONS_PATH", str(positions_log_path))

    for handler in list(trade_ledger_module.trade_logger.handlers):
        trade_ledger_module.trade_logger.removeHandler(handler)

    trade_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    trade_handler.setFormatter(logging.Formatter("%(message)s"))
    trade_ledger_module.trade_logger.addHandler(trade_handler)
    trade_ledger_module.trade_logger.setLevel(logging.INFO)
    trade_ledger_module.trade_logger.propagate = False

    def _restore_logger() -> None:
        for handler in list(trade_ledger_module.trade_logger.handlers):
            trade_ledger_module.trade_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        for handler in original_handlers:
            trade_ledger_module.trade_logger.addHandler(handler)
        trade_ledger_module.trade_logger.setLevel(logging.INFO)
        trade_ledger_module.trade_logger.propagate = False
        trade_ledger_module.TRADES_LOG_PATH = original_trades_log_path
        trade_ledger_module.POSITIONS_PATH = original_positions_path

    request.addfinalizer(_restore_logger)

    position_manager = PositionManager()
    ledger = TradeLedger(position_manager)

    trade_id = "integration-trade"
    pair = "BTC/USDC"
    entry_price = 100.0
    trade_size = 0.01
    timestamp = datetime.now(timezone.utc).isoformat()

    ledger.log_trade(
        trading_pair=pair,
        trade_size=trade_size,
        strategy_name="IntegrationStrategy",
        trade_id=trade_id,
        strategy_instance=None,
        confidence=0.8,
        entry_price=entry_price,
        regime="test",
        capital_buffer=0.5,
        rsi=50.0,
        adx=25.0,
    )
    position_manager.open_position(
        trade_id=trade_id,
        pair=pair,
        size=trade_size,
        entry_price=entry_price,
        strategy="IntegrationStrategy",
        confidence=0.8,
        timestamp=timestamp,
    )

    exit_price = entry_price * 1.05
    exits = position_manager.check_exits(
        {pair: exit_price},
        tp=0.01,
        sl=0.05,
        trailing_stop=0.05,
        max_hold_bars=1,
    )
    assert exits, "Expected exit trigger to close integration position"

    for exit_trade_id, price, reason in exits:
        ledger.update_trade(trade_id=exit_trade_id, exit_price=price, reason=reason)

    assert not position_manager.positions, "Position manager should clear closed trades"

    metrics = learning_machine.run_learning_cycle()
    assert metrics["total_trades"] == 1
    assert metrics["wins"] == 1
    assert metrics["losses"] == 0

    suggestions_written = learning_machine.run_learning_machine()
    assert os.path.exists("logs/learning_feedback.jsonl")

    with open(log_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert any(rec.get("status") == "closed" for rec in records)
    assert suggestions_written >= 0
