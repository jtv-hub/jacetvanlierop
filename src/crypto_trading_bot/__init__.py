"""Crypto Trading Bot package namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "trading_logic",
    "portfolio_state",
    "simulation",
    "strategies",
    "scheduler",
    "market_data",
    "confidence_analytics",
]

if TYPE_CHECKING:
    from . import (
        confidence_analytics,
        market_data,
        portfolio_state,
        scheduler,
        simulation,
        strategies,
        trading_logic,
    )  # noqa: F401
