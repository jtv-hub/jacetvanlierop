"""
Gatekeeper module for final decision-making on learning suggestions.
Evaluates shadow test results and approves/rejects changes.
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# === Setup rotating logger ===
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("gatekeeper")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "logs/gatekeeper.log", maxBytes=50 * 1024 * 1024, backupCount=5  # 50 MB
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# === File paths ===
RESULTS_FILE = "learning/shadow_test_results.jsonl"
DECISIONS_FILE = "learning/final_decisions.jsonl"


def evaluate():
    """Evaluate shadow test results and finalize gatekeeper decisions."""
    if not os.path.exists(RESULTS_FILE):
        logger.warning("No shadow test results found at %s", RESULTS_FILE)
        return

    final_decisions = []
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    result = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Apply decision rules
                confidence = result.get("confidence", 0)
                status = result.get("status", "fail")

                decision = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "strategy_name": result.get("strategy_name"),
                    "param_change": result.get("param_change"),
                    "confidence": confidence,
                    "shadow_test_status": status,
                    "final_decision": (
                        "approved"
                        if status == "pass" and confidence >= 0.7
                        else "rejected"
                    ),
                }

                final_decisions.append(decision)

                logger.info(
                    "Decision: %s | Strategy=%s | Confidence=%.2f",
                    decision["final_decision"],
                    decision["strategy_name"],
                    confidence,
                )

    except OSError as e:
        logger.error("Error reading shadow test results: %s", e)
        return

    # Write all decisions to file
    try:
        os.makedirs("learning", exist_ok=True)
        with open(DECISIONS_FILE, "a", encoding="utf-8") as outfile:
            for d in final_decisions:
                outfile.write(json.dumps(d) + "\n")
    except OSError as e:
        logger.error("Error writing decisions: %s", e)

    logger.info("✅ Gatekeeper evaluation completed.")


def run() -> None:
    """Entry point wrapper for pipeline usage."""
    evaluate()


if __name__ == "__main__":
    evaluate()
