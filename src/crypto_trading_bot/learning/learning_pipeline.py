"""
Learning Pipeline Controller
Coordinates the review-learning-shadow process into a single run.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from crypto_trading_bot.learning.learning_machine import run_learning_machine
from crypto_trading_bot.learning.review_learning_ledger import review_ledger
from crypto_trading_bot.learning.shadow_test_runner import run_shadow_tests

# === Setup rotating logger ===
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("learning_pipeline")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "logs/learning_pipeline.log",
    maxBytes=50 * 1024 * 1024,  # 50 MB
    backupCount=3,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

__all__ = ["run_learning_pipeline"]


def run_learning_pipeline() -> None:
    """Run the consolidated learning pipeline using canonical log files.

    Flow:
    1) Generate learning suggestions -> logs/learning_feedback.jsonl
    2) Fallback: generate review-based suggestions if LM produced none
    3) Run shadow tests on suggestions -> logs/shadow_test_results.jsonl
    """
    logger.info("🚀 Starting learning pipeline")

    # Step 1: primary suggestion source
    wrote = 0
    try:
        wrote = run_learning_machine()  # writes to logs/learning_feedback.jsonl
        logger.info("Learning machine wrote %d suggestion(s)", wrote)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("run_learning_machine failed: %s", exc, exc_info=True)

    # Step 1b: fallback suggestions from ledger review if none
    if wrote == 0:
        try:
            review_ledger()  # appends to logs/learning_feedback.jsonl in canonical format
            logger.info("Fallback review_ledger executed (no LM suggestions)")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("review_ledger fallback failed: %s", exc, exc_info=True)

    # Step 2: run shadow tests against canonical suggestions file
    try:
        run_shadow_tests(
            input_file="logs/learning_feedback.jsonl",
            output_file="logs/shadow_test_results.jsonl",
        )
        logger.info("✅ Learning pipeline completed successfully")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Shadow tests failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    run_learning_pipeline()
