"""Project-wide constant values for configuration paths."""

from __future__ import annotations

from pathlib import Path

# Path to the kill-switch sentinel used to halt live trading.
KILL_SWITCH_FILE = Path("config/kill_switch.json")

__all__ = ["KILL_SWITCH_FILE"]
