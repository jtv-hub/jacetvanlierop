"""Crypto Trading Bot package namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["bot", "ledger", "scripts", "utils"]

if TYPE_CHECKING:
    from . import bot, ledger, scripts, utils  # noqa: F401
