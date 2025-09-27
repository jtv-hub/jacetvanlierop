#!/usr/bin/env python3
"""Demonstrate ledger reload/alert/pause behaviour for missing trades.

This script simulates a trade lifecycle using a temporary log directory. It:

1. Logs a trade via the ledger and records an open position.
2. Deletes the trade from the backing ``trades.log`` file to simulate manual
   corruption or truncation.
3. Calls ``update_trade`` which now reloads the log, detects the missing entry,
   emits a CRITICAL alert, and requests a one-cycle pause for new trades.

The script prints a compact summary so operators can visually confirm the
fail-safe paths without disturbing production log files.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch

try:
    from crypto_trading_bot.ledger import trade_ledger as ledger_module
except ModuleNotFoundError:  # pragma: no cover - fallback when running from repo root
    REPO_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = REPO_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from crypto_trading_bot.ledger import trade_ledger as ledger_module


class _DummyPositionManager:
    """Minimal stub exposing the interface used by TradeLedger."""

    def __init__(self) -> None:
        self.positions: Dict[str, Dict[str, object]] = {}

    def load_positions_from_file(self) -> None:  # pragma: no cover - simple stub
        return None


def _rebind_logger(logger: logging.Logger, handler_path: Path, fmt: str) -> None:
    """Attach a single FileHandler pointing at ``handler_path``."""

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(handler_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def _summarize_alert(alert_args: Tuple, alert_kwargs: Dict[str, object]) -> str:
    message = str(alert_args[0]) if alert_args else "<no message>"
    level = alert_kwargs.get("level", "INFO")
    context = alert_kwargs.get("context") or {}
    return f"level={level} message={message} context={json.dumps(context, default=str)}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        logs_dir = base / "logs"
        trades_path = logs_dir / "trades.log"
        positions_path = logs_dir / "positions.jsonl"
        system_log_path = logs_dir / "system.log"

        with patch.multiple(
            ledger_module,
            TRADES_LOG_PATH=str(trades_path),
            POSITIONS_PATH=str(positions_path),
            SYSTEM_LOG_PATH=str(system_log_path),
        ):
            _rebind_logger(ledger_module.trade_logger, trades_path, "%(message)s")
            _rebind_logger(
                ledger_module.system_logger,
                system_log_path,
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
            )

            pm = _DummyPositionManager()
            ledger = ledger_module.TradeLedger(pm)

            trade_id = ledger.log_trade(
                "BTC/USD",
                trade_size=0.01,
                strategy_name="DemoStrategy",
                confidence=0.75,
                entry_price=20_000.0,
            )
            pm.positions[trade_id] = {
                "pair": "BTC/USD",
                "size": 0.01,
                "strategy": "DemoStrategy",
                "entry_price": 20_000.0,
            }
            ledger.open_position(trade_id, pm.positions[trade_id])

            # Corrupt the trades log by removing the entry entirely.
            filtered_lines = []
            for line in trades_path.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("trade_id") == trade_id:
                        continue
                except json.JSONDecodeError:
                    pass
                filtered_lines.append(line)
            trades_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

            # Drop in-memory references so update must rely on reload.
            ledger.trades.clear()
            ledger.trade_index.clear()

            with patch("crypto_trading_bot.ledger.trade_ledger.send_alert") as mock_alert:
                ledger.update_trade(trade_id, exit_price=21_000.0, reason="demo_exit")
                pause_requested, pause_reason = ledger.consume_pause_request()

            print("=== Missing Trade Demo ===")
            print(f"trade_id           : {trade_id}")
            print(f"alert_invoked      : {mock_alert.called}")
            if mock_alert.called:
                summary = _summarize_alert(*mock_alert.call_args)
                print(f"alert_payload      : {summary}")
            print(f"pause_requested    : {pause_requested}")
            print(f"pause_reason       : {pause_reason}")
            print(f"trades.log entries : {trades_path.read_text(encoding='utf-8').strip() or '<empty>'}")
            print(f"system.log path    : {system_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
