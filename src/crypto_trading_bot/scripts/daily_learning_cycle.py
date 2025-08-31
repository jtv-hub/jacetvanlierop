# src/crypto_trading_bot/scripts/daily_learning_cycle.py

"""
Daily Learning Cycle
Integrates ledger -> learning machine -> report export.
Intended to be scheduled (daily/weekly).
"""

import logging

from crypto_trading_bot.learning.learning_machine import LearningMachine
from crypto_trading_bot.scripts.export_learning_report import export_learning_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_daily_learning_cycle():
    """Run full learning cycle: ledger -> learning -> report export."""
    logger.info("Starting daily learning cycle...")

    # Initialize learning machine
    lm = LearningMachine()

    # Generate and log report
    report = lm.generate_report()
    logger.info("Learning Analysis Report:")
    for k, v in report.items():
        logger.info("  %s: %s", k, v)

    # Export report to JSON/CSV
    export_learning_report()

    logger.info("Daily learning cycle complete.")


if __name__ == "__main__":
    run_daily_learning_cycle()
