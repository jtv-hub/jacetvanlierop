# src/crypto_trading_bot/scripts/bootstrap_learning.py

"""
Bootstrap script for the Learning Machine.

This script initializes the LearningMachine and runs a simple
confidence-processing test with dummy trade data. Its purpose is
to verify imports, logging, and end-to-end functionality.
"""

import logging

from crypto_trading_bot.learning.learning_machine import LearningMachine

logger = logging.getLogger(__name__)


class BootstrapLearning:
    """
    A test harness for the LearningMachine.
    It feeds dummy trades into the process_trade_confidence method.
    """

    def __init__(self):
        self.lm = LearningMachine()

    def run_test(self):
        """
        Run a simple test of the LearningMachine with dummy trade data.
        """
        dummy_trade = {
            "strategy_name": "TestStrategy",
            "confidence_score": 0.85,
            "regime": "trending",
            "roi": 2.5,
            "pnl": 150.0,
            "win": True,
            "trade_duration": 42.0,
            "vol_entry": 0.012,
            "vol_exit": 0.010,
        }

        logger.info("Sending dummy trade to LearningMachine...")
        self.lm.process_trade_confidence(
            dummy_trade["strategy_name"],
            dummy_trade["confidence_score"],
            dummy_trade["regime"],
            dummy_trade["roi"],
            dummy_trade["pnl"],
            dummy_trade["win"],
            dummy_trade["trade_duration"],
            dummy_trade["vol_entry"],
            dummy_trade["vol_exit"],
        )
        logger.info("Dummy trade processed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tester = BootstrapLearning()
    tester.run_test()
