"""Compatibility shim to expose the src/crypto_trading_bot package."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _PROJECT_ROOT / "src" / "crypto_trading_bot"

if _SRC_PACKAGE.exists():
    _pkg_path = str(_SRC_PACKAGE)
    if _pkg_path not in __path__:
        __path__.insert(0, _pkg_path)
