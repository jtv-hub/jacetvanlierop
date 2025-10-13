"""Shared configuration constants for the crypto trading bot."""

from __future__ import annotations

import os
from pathlib import Path

# --- Live trading safety thresholds ---

DEFAULT_RISK_FAILURE_LIMIT: int = 5
DEFAULT_RISK_DRAWDOWN_THRESHOLD: float = 0.10

# --- Auto-pause heuristics ---

DEFAULT_AUTO_PAUSE_MAX_DRAWDOWN: float = 0.10
DEFAULT_AUTO_PAUSE_MAX_CONSEC_LOSSES: int = 5
DEFAULT_AUTO_PAUSE_TOTAL_ROI: float = -0.15

# --- Deployment controls ---

DEFAULT_CANARY_MAX_FRACTION: float = 0.05

# --- Safety sentinels ---

KILL_SWITCH_FILE = Path(os.getenv("CRYPTO_TRADING_BOT_KILL_SWITCH_FILE", "logs/kill_switch.flag")).expanduser()

__all__ = [
    "DEFAULT_AUTO_PAUSE_MAX_CONSEC_LOSSES",
    "DEFAULT_AUTO_PAUSE_MAX_DRAWDOWN",
    "DEFAULT_AUTO_PAUSE_TOTAL_ROI",
    "DEFAULT_CANARY_MAX_FRACTION",
    "DEFAULT_RISK_DRAWDOWN_THRESHOLD",
    "DEFAULT_RISK_FAILURE_LIMIT",
    "KILL_SWITCH_FILE",
]
