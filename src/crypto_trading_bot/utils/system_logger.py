"""Shared system logger utilities."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

SYSTEM_LOG_PATH = Path("logs/system.log")
_DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"


def get_system_logger(name: str = "trade_ledger.system") -> logging.Logger:
    """Return a configured RotatingFile logger for system diagnostics."""

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    SYSTEM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        SYSTEM_LOG_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if _DEBUG_MODE else logging.INFO)
    logger.propagate = False
    return logger
