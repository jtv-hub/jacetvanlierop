"""
Logging setup for the crypto trading bot.
Handles file and console log formatting and routing.
"""

import logging
import os

from crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler


def setup_logging():
    """
    Sets up logging configuration for the bot.
    """
    os.makedirs("logs", exist_ok=True)

    rotating_handler = get_rotating_handler("heartbeat.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[rotating_handler, logging.StreamHandler()],
    )
