"""
Learning Pipeline Controller
Coordinates the review-learning-shadow process into a single run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.learning import learning_machine as lm
from crypto_trading_bot.learning import review_learning_ledger as review
from crypto_trading_bot.learning import shadow_test_runner as shadow
from crypto_trading_bot.optimizer.genetic_optimizer import run_nsga3_evolution
from crypto_trading_bot.utils.kraken_client import get_usdc_balance
from crypto_trading_bot.utils.sqlite_logger import process_trade_confidence

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
handler.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(handler)

# === File paths ===
SUGGESTIONS_FILE = "learning/suggestions.jsonl"
RESULTS_FILE = "reports/shadow_test_results.jsonl"

__all__ = ["run_learning_pipeline"]  # makes Pylint happy


def _count_json_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as infile:
            return sum(1 for line in infile if line.strip())
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return 0


def _load_shadow_results(path: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return results
    try:
        with open(path, "r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logger.error("Invalid JSON line in results: %s", line.strip())
    except OSError as exc:
        logger.error("Failed to read shadow test results: %s", exc)
    return results


def run_learning_pipeline(*, debug: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """
    Run the full learning pipeline:
    1. Review trade ledger and generate suggestions
    2. Run shadow tests on those suggestions
    3. Update learning machine with shadow test results
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    logger.info("🚀 Starting learning pipeline (debug=%s dry_run=%s)", debug, dry_run)

    balance = get_usdc_balance()
    if not balance or balance <= 0:
        logger.error("KRAKEN: No USDC balance — aborting")
        return {
            "status": "aborted",
            "reason": "no_balance",
            "suggestions": 0,
            "passed": 0,
            "failed": 0,
            "trades": _count_json_lines("logs/trades.log"),
        }

    total_trades = _count_json_lines("logs/trades.log")
    logger.debug("Detected %d trade entries in ledger", total_trades)

    if dry_run:
        logger.info("[dry-run] Skipping ledger review step")
    else:
        try:
            logger.info("Step 1: Reviewing trade ledger for suggestions...")
            review.run()
            logger.info("Review complete")
        except Exception:  # noqa: BLE001 - guard the nightly pipeline
            logger.error("Review failed", exc_info=True)
            return {
                "status": "failed",
                "reason": "review_failed",
                "suggestions": 0,
                "passed": 0,
                "failed": 0,
                "trades": total_trades,
            }

    suggestion_count = _count_json_lines(SUGGESTIONS_FILE)
    logger.info("Generated %d suggestion(s).", suggestion_count)

    if dry_run:
        logger.info("[dry-run] Skipping shadow test execution")
    else:
        try:
            logger.info("Step 2: Running shadow tests...")
            shadow.run_shadow_tests()
            logger.info("Shadow tests complete")
        except Exception:  # noqa: BLE001 - guard the nightly pipeline
            logger.error("Shadow tests failed", exc_info=True)
            return {
                "status": "failed",
                "reason": "shadow_failed",
                "suggestions": suggestion_count,
                "passed": 0,
                "failed": 0,
                "trades": total_trades,
            }

    results = _load_shadow_results(RESULTS_FILE)
    if not results:
        logger.warning("No shadow test results found. Exiting pipeline.")
        return {
            "status": "no_results",
            "reason": "no_shadow_results",
            "suggestions": suggestion_count,
            "passed": 0,
            "failed": 0,
            "trades": total_trades,
        }

    passed = [r for r in results if r.get("status") == "pass"]
    failed = [r for r in results if r.get("status") == "fail"]
    logger.info("Shadow tests complete: %d passed, %d failed", len(passed), len(failed))

    if dry_run:
        logger.info("[dry-run] Skipping learning machine update for %d passed result(s)", len(passed))
    else:
        try:
            logger.info("Step 3: Updating learning machine with results...")
            for result in passed:
                strategy = result.get("strategy_name", "unknown")
                conf = result.get("confidence", 0.0)
                regime = result.get("regime", "unknown")
                lm.process_trade_confidence(strategy, conf, regime)
                try:
                    process_trade_confidence(
                        strategy=strategy,
                        confidence=float(conf or 0.0),
                        regime=regime,
                        status="recorded",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to log confidence to SQLite: %s", exc)
            logger.info("Learning machine update complete")
        except Exception:  # noqa: BLE001 - guard the nightly pipeline
            logger.error("Learning machine update failed", exc_info=True)
            return {
                "status": "failed",
                "reason": "learning_update_failed",
                "suggestions": suggestion_count,
                "passed": len(passed),
                "failed": len(failed),
                "trades": total_trades,
            }

    try:
        min_shadow_trades = int(CONFIG.get("min_shadow_trades", 5))
    except NameError:
        min_shadow_trades = 5
        logger.warning("CONFIG unavailable, defaulting min_shadow_trades=5")
    win_rate = len(passed) / len(results) if results else 0.0
    logger.info("Shadow win rate: %.1f%%", win_rate * 100)

    if len(results) < min_shadow_trades:
        logger.info(
            "Only %d shadow test(s) available — need ≥%d to trigger NSGA-III. Skipping evolution.",
            len(results),
            min_shadow_trades,
        )
    elif win_rate >= 0.5:
        if dry_run:
            logger.info("[dry-run] Shadow win rate ≥50%% → would launch NSGA-III (generations=30)")
        else:
            logger.info("Shadow win rate ≥50%% → launching NSGA-III")
            try:
                run_nsga3_evolution(generations=30)
            except Exception:  # noqa: BLE001 - optimisation failure isn't fatal
                logger.error("NSGA-III failed", exc_info=True)
    else:
        logger.info("Shadow win rate <50%% — skipping NSGA-III")

    metrics = {
        "status": "completed",
        "reason": None,
        "suggestions": suggestion_count,
        "passed": len(passed),
        "failed": len(failed),
        "trades": total_trades,
    }
    logger.info(
        "Pipeline complete: %d trades, %d suggestions, %d passed, %d failed",
        metrics["trades"],
        metrics["suggestions"],
        metrics["passed"],
        metrics["failed"],
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the learning pipeline.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("--dry-run", action="store_true", help="Skip actions that mutate state.")
    cli_args = parser.parse_args()
    run_learning_pipeline(debug=cli_args.debug, dry_run=cli_args.dry_run)
