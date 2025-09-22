"""Configuration loader for crypto_trading_bot.

Loads environment variables from a `.env` file (project root) and exposes
application configuration via the `CONFIG` dictionary.
"""

import logging
import os

from dotenv import load_dotenv

LIVE_MODE_LABEL = "\U0001f6a8 LIVE MODE \U0001f6a8"
PAPER_MODE_LABEL = "PAPER MODE"
is_live: bool = False


def set_live_mode(flag: bool) -> None:
    """Set the global live-trading toggle."""

    global is_live  # noqa: PLW0603 - intentional shared toggle
    is_live = bool(flag)


def get_mode_label() -> str:
    """Return a human-readable label for the current trading mode."""

    return LIVE_MODE_LABEL if is_live else PAPER_MODE_LABEL


def _to_float(value: str | None, default: float) -> float:
    """Best-effort float conversion with a fallback default."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str | None, default: int) -> int:
    """Best-effort int conversion with a fallback default."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Initialize logger for this module
logger = logging.getLogger(__name__)

# Load environment variables from a .env file in the project root
load_dotenv()


_env_live_flag = os.getenv("CRYPTO_TRADING_BOT_LIVE")
if _env_live_flag is not None:
    normalized = _env_live_flag.strip().lower()
    set_live_mode(normalized in {"1", "true", "yes", "on"})

    if logger.isEnabledFor(logging.INFO):
        logger.info("Live trading mode overridden by env: %s -> %s", _env_live_flag, is_live)

CONFIG: dict = {
    # Centralized list of tradable pairs used across the app.
    # Added to ensure no hardcoded pairs scattered in the codebase.
    # Ordering matches requested scan order.
    "tradable_pairs": [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "LINK/USD",
    ],
    "rsi": {
        "period": 14,
        "lower": 48,
        "upper": 75,
        # Lower exit threshold for easier testing; tune in paper/live
        "exit_upper": 55,
    },
    "max_portfolio_risk": 0.10,
    "min_volume": 100,
    "trade_size": {"min": 0.001, "max": 0.005},
    "slippage": {
        "majors": 0.001,  # 0.1%
        "alts_min": 0.005,  # 0.5%
        "alts_max": 0.01,  # 1.0%
        "use_random": False,
    },
    "buffer_defaults": {
        "trending": 1.0,
        "chop": 0.5,
        "volatile": 0.5,
        "flat": 0.25,
        "unknown": 0.25,
    },
    "correlation": {"window": 30, "threshold": 0.7},
    "paper_mode": {
        "starting_balance": _to_float(
            os.getenv("PAPER_STARTING_BALANCE"),
            100_000.0,
        )
    },
    "live_mode": {
        "balance_env_var": os.getenv(
            "CRYPTO_TRADING_BOT_BALANCE_ENV",
            "CRYPTO_TRADING_BOT_LIVE_BALANCE",
        ),
        "fallback_balance": _to_float(
            os.getenv("LIVE_BALANCE_FALLBACK"),
            0.0,
        ),
        "balance_provider": os.getenv("ACCOUNT_BALANCE_PROVIDER"),
    },
    "auto_pause": {
        "max_drawdown_pct": _to_float(
            os.getenv("AUTO_PAUSE_MAX_DRAWDOWN"),
            0.10,
        ),
        "max_consecutive_losses": _to_int(
            os.getenv("AUTO_PAUSE_MAX_CONSEC_LOSSES"),
            5,
        ),
        "max_total_roi_pct": _to_float(
            os.getenv("AUTO_PAUSE_MAX_TOTAL_ROI"),
            -0.15,
        ),
    },
    # API credentials loaded from environment
    "kraken_api_key": os.getenv("KRAKEN_API_KEY"),
    "kraken_api_secret": os.getenv("KRAKEN_API_SECRET"),
}

# Safe debug log showing whether credentials are present (not the values)
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(
        "Config: kraken_api_key_present=%s kraken_api_secret_present=%s",
        bool(CONFIG.get("kraken_api_key")),
        bool(CONFIG.get("kraken_api_secret")),
    )


__all__ = [
    "CONFIG",
    "LIVE_MODE_LABEL",
    "PAPER_MODE_LABEL",
    "get_mode_label",
    "is_live",
    "set_live_mode",
]
