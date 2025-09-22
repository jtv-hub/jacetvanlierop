"""Global runtime toggles for live trading safety."""

from __future__ import annotations

is_live: bool = False  # Set to True ONLY when ready for live trading

LIVE_MODE_LABEL = "\U0001F6A8 LIVE MODE \U0001F6A8"
PAPER_MODE_LABEL = "PAPER MODE"


def get_mode_label() -> str:
    """Return a human-readable label for the current trading mode."""
    return LIVE_MODE_LABEL if is_live else PAPER_MODE_LABEL


__all__ = ["is_live", "get_mode_label", "LIVE_MODE_LABEL", "PAPER_MODE_LABEL"]
