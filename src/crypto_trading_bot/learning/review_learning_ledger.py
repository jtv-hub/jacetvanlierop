"""
Review Learning Ledger
Analyzes trade logs and generates learning suggestions.
"""

import logging
import json
import os

logger = logging.getLogger("review_learning_ledger")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def review_ledger():
    """
    Core function to read the ledger and generate learning suggestions.
    This is where the main logic goes.
    """
    logger.info("[review] Starting ledger review...")

    ledger_file = "logs/trades.log"
    suggestions_file = "learning/suggestions.jsonl"

    if not os.path.exists(ledger_file):
        logger.warning("[review] Ledger file not found: %s", ledger_file)
        return

    try:
        with open(ledger_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        logger.error("[review] Failed to read ledger: %s", e)
        return

    suggestions = []
    for line in lines:
        try:
            trade = json.loads(line)
            # --- Example: basic suggestion logic ---
            if trade.get("status") == "loss":
                suggestions.append(
                    {
                        "pair": trade.get("pair"),
                        "strategy": trade.get("strategy"),
                        "action": "adjust_parameters",
                        "confidence": 0.6,
                    }
                )
        except json.JSONDecodeError:
            logger.warning("[review] Skipping invalid log line: %s", line.strip())

    if suggestions:
        try:
            with open(suggestions_file, "a", encoding="utf-8") as f:
                for s in suggestions:
                    f.write(json.dumps(s) + "\n")
            logger.info(
                "[review] Wrote %d suggestions to %s",
                len(suggestions),
                suggestions_file,
            )
        except OSError as e:
            logger.error("[review] Failed to write suggestions: %s", e)
    else:
        logger.info("[review] No suggestions generated.")


def run():
    """
    Entrypoint so nightly_pipeline can call review_learning_ledger.run()
    """
    review_ledger()


if __name__ == "__main__":
    run()
