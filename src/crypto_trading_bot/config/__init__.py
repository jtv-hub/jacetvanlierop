"""Configuration loader for crypto_trading_bot.

Loads environment variables from a `.env` file (project root) and exposes
application configuration via the `CONFIG` dictionary.
"""

import logging
import os

from dotenv import load_dotenv

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Load environment variables from a .env file in the project root
load_dotenv()

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
