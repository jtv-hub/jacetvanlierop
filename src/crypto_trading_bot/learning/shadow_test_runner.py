"""
Shadow Test Runner
Runs shadow tests on strategy suggestions and logs results.
"""

import json
import logging
import os
import statistics
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

# === Setup rotating logger ===
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("shadow_test_runner")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler("logs/shadow_test_runner.log", maxBytes=50 * 1024 * 1024, backupCount=3)  # 50 MB
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# === File paths ===
SUGGESTIONS_FILE = "reports/suggestions_latest.json"
RESULTS_FILE = "reports/shadow_test_results.jsonl"


def run_shadow_tests(input_file: str = SUGGESTIONS_FILE, output_file: str = RESULTS_FILE):
    """
    Run shadow tests on suggestions from the learning engine.
    Produces results with adaptive pass/fail threshold and audit trail.
    """
    if not os.path.exists(input_file):
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        with open(input_file, "w", encoding="utf-8"):
            pass
        logger.info(
            "Created empty suggestions file at %s. " "Please run review_learning_ledger.py to add suggestions.",
            input_file,
        )
        return

    results = []
    confidences = []

    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            content = infile.read().strip()
            if content:
                try:
                    suggestions = json.loads(content)
                    if isinstance(suggestions, dict):
                        suggestions = [suggestions]
                    elif not isinstance(suggestions, list):
                        raise ValueError("Invalid JSON structure")
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in %s: %s", input_file, e)
                    return
            else:
                suggestions = []

        if not suggestions:
            logger.warning("No valid suggestions found in %s. Skipping shadow tests.", input_file)
            return

        # === Determine adaptive threshold ===
        for suggestion in suggestions:
            conf = suggestion.get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(conf)

        threshold = 0.5
        if confidences and len(confidences) > 1:
            median_conf = statistics.median(confidences)
            threshold = max(0.5, median_conf)
        logger.info(
            "Adaptive threshold set to %.2f based on median confidence",
            threshold,
        )

        # === Evaluate each suggestion ===
        for suggestion in suggestions:
            confidence = suggestion.get("confidence", 0)
            reason_str = suggestion.get("reason", "")

            def extract_metric(label, reason=reason_str):
                try:
                    part = reason.split(f"{label}=")[1]
                    return float(part.split(",")[0] if "," in part else part)
                except (IndexError, ValueError):
                    return 0

            avg_roi = extract_metric("ROI")
            win_rate = extract_metric("Win Rate")
            sharpe = extract_metric("Sharpe")

            status = "pass" if confidence >= threshold else "fail"
            reason = (
                f"Confidence {confidence:.2f} below threshold {threshold:.2f}"
                if status == "fail"
                else "Met confidence threshold"
            )
            logger.info(
                "Suggestion %s: %s (confidence: %.2f, threshold: %.2f)",
                suggestion.get("suggestion", "unknown"),
                status,
                confidence,
                threshold,
            )

            result = {
                "shadow_test_id": str(uuid.uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "strategy_name": suggestion.get("strategy_name", "unknown"),
                "param_change": suggestion.get("param_change", {}),
                "rationale": suggestion.get("rationale", ""),
                "confidence": confidence,
                "avg_roi": avg_roi,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "threshold": threshold,
                "status": status,
                "reason": reason,
            }
            results.append(result)

    except OSError as e:
        logger.error("Error reading suggestions from %s: %s", input_file, e)
        return
    except (ValueError, TypeError, KeyError) as e:
        logger.error("Unexpected error in %s: %s", input_file, e)
        return

    # === Write results ===
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as outfile:
            for r in results:
                outfile.write(json.dumps(r) + "\n")
        logger.info("✅ Shadow test results saved to %s", output_file)
    except OSError as e:
        logger.error("Failed to write results to %s: %s", output_file, e)


if __name__ == "__main__":
    run_shadow_tests()
