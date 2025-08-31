"""
Logging setup for the crypto trading bot.
Handles file and console log formatting and routing.
"""

import logging
import os


def setup_logging():
    """
    Sets up logging configuration for the bot.
    """
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/heartbeat.log"), logging.StreamHandler()],
    )
