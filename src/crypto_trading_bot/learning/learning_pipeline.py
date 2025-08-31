"""
Learning Pipeline Controller
Coordinates the review-learning-shadow process into a single run.
"""

import importlib
import json
import logging
import os
from logging.handlers import RotatingFileHandler

# === Dynamic imports (fixes IDE & runtime issues) ===
review = importlib.import_module("crypto_trading_bot.learning.review_learning_ledger")
shadow = importlib.import_module("crypto_trading_bot.learning.shadow_test_runner")
lm = importlib.import_module("crypto_trading_bot.learning.learning_machine")

# === Setup rotating logger ===
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("learning_pipeline")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "logs/learning_pipeline.log",
    maxBytes=50 * 1024 * 1024,  # 50 MB
    backupCount=3,
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# === File paths ===
SUGGESTIONS_FILE = "learning/suggestions.jsonl"
RESULTS_FILE = "learning/shadow_test_results.jsonl"

__all__ = ["run_learning_pipeline"]  # makes Pylint happy


def run_learning_pipeline():
    """
    Run the full learning pipeline:
    1. Review trade ledger and generate suggestions
    2. Run shadow tests on those suggestions
    3. Update learning machine with shadow test results
    """
    logger.info("🚀 Starting learning pipeline")

    try:
        # === Step 1: Review ledger and generate suggestions ===
        logger.info("Step 1: Reviewing trade ledger for suggestions...")
        review.main()

        # Count how many suggestions were generated
        suggestion_count = 0
        if os.path.exists(SUGGESTIONS_FILE):
            with open(SUGGESTIONS_FILE, "r", encoding="utf-8") as infile:
                suggestion_count = sum(1 for _ in infile)
        logger.info("Generated %d suggestion(s).", suggestion_count)

        # === Step 2: Run shadow tests ===
        logger.info("Step 2: Running shadow tests...")
        shadow.run_shadow_tests()

        # Collect shadow test results
        results = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        results.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON line in results: %s", line.strip())

        if not results:
            logger.warning("No shadow test results found. Exiting pipeline.")
            return

        passed = [r for r in results if r.get("status") == "pass"]
        failed = [r for r in results if r.get("status") == "fail"]
        logger.info("Shadow tests complete: %d passed, %d failed", len(passed), len(failed))

        # === Step 3: Update learning machine ===
        logger.info("Step 3: Updating learning machine with results...")
        for result in passed:
            strategy = result.get("strategy_name", "unknown")
            conf = result.get("confidence", 0.0)
            regime = result.get("regime", "unknown")
            lm.process_trade_confidence(strategy, conf, regime)

        logger.info("✅ Learning pipeline completed successfully")

    except (OSError, ValueError, RuntimeError) as e:
        logger.error("❌ Learning pipeline failed: %s", e, exc_info=True)


if __name__ == "__main__":
    run_learning_pipeline()
