"""Lazy accessors for bot submodules to avoid circular imports."""

from __future__ import annotations

import importlib

__all__ = ["market_data", "trading_logic"]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
