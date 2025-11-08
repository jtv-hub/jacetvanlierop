"""Shared access helpers for the global trade ledger instance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from crypto_trading_bot.ledger.trade_ledger import TradeLedger

_LEDGER_SINGLETON = None


def get_ledger(position_manager: Any | None = None) -> "TradeLedger":
    """Return the shared ``TradeLedger`` instance, initialising it lazily."""

    from crypto_trading_bot.ledger.trade_ledger import TradeLedger  # local import

    global _LEDGER_SINGLETON  # pylint: disable=global-statement

    if _LEDGER_SINGLETON is None:
        if position_manager is None:
            from crypto_trading_bot.bot import trading_logic

            position_manager = trading_logic.position_manager

        _LEDGER_SINGLETON = TradeLedger(position_manager)

    return _LEDGER_SINGLETON


__all__ = ["get_ledger"]
