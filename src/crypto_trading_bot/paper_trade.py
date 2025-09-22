"""
Paper trading script for smoke testing.
Runs a simple loop to simulate trades and log results.
"""

import logging
import time

from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.config import CONFIG, get_mode_label, is_live

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def paper_trade(iterations: int = 5, interval: int = 10):
    """Run a simple paper trading loop for smoke testing."""
    tradable_pairs = CONFIG.get("tradable_pairs", [])
    logging.info("Starting paper trading across pairs: %s", tradable_pairs)
    logging.info("Mode: %s (is_live=%s)", get_mode_label(), is_live)

    for i in range(iterations):
        logging.info("Iteration %s/%s", i + 1, iterations)

        # Evaluate signals across the centralized pair list
        evaluate_signals_and_trade(tradable_pairs=tradable_pairs)
        logging.info("✅ evaluate_signals_and_trade completed for all pairs")

        time.sleep(interval)

    logging.info("Paper trading complete.")


if __name__ == "__main__":
    paper_trade()
