"""
Paper trading script for smoke testing.
Runs a simple loop to simulate trades and log results.
"""

import time
import logging

from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def paper_trade(iterations: int = 5, interval: int = 10):
    """Run a simple paper trading loop for smoke testing."""
    trading_pair = "BTC/USD"

    logging.info("Starting paper trading for %s...", trading_pair)

    for i in range(iterations):
        logging.info("Iteration %s/%s", i + 1, iterations)

        # Evaluate signals and/or force trade
        evaluate_signals_and_trade()

        logging.info("✅ evaluate_signals_and_trade triggered for %s", trading_pair)

        time.sleep(interval)

    logging.info("Paper trading complete.")


if __name__ == "__main__":
    paper_trade()
