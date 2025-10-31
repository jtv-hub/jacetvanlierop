"""Shared access helpers for the global trade ledger instance."""

from __future__ import annotations

from typing import Any

from crypto_trading_bot.ledger.trade_ledger import TradeLedger

_LEDGER: TradeLedger | None = None


def get_ledger(position_manager: Any | None = None) -> TradeLedger:
    """Return the shared ``TradeLedger`` instance, initialising it lazily."""

    global _LEDGER

    if _LEDGER is None:
        if position_manager is None:
            from crypto_trading_bot import trading_logic

            position_manager = trading_logic.position_manager

        _LEDGER = TradeLedger(position_manager)

    return _LEDGER


__all__ = ["get_ledger"]
