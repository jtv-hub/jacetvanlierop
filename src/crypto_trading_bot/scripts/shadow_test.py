"""
Shadow testing engine for learning proposals.

Simulates strategy modifications on historical data to compare
original vs proposed performance.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LEDGER_PATH = Path("logs/learning_ledger.jsonl")

# Simulated suggestion: Change RSI buy threshold from 30 → 40
SUGGESTED_RSI_THRESHOLD = 40


def load_ledger():
    """Load all valid trade entries from the learning ledger."""
    if not LEDGER_PATH.exists():
        logging.warning("Ledger file not found.")
        return []

    entries = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as file:
        for line in file:
            try:
                entry = json.loads(line)
                if entry.get("signal") == "buy" and entry.get("confidence") is not None:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def simulate_rsi_threshold(entries, new_threshold: float):
    """
    Simulate what would have happened if RSI threshold had been higher.
    Assumes lower RSI → higher likelihood of signal.
    """
    original_confidences = []
    shadow_confidences = []

    for entry in entries:
        rsi_value = entry.get("rsi") or entry.get("rsi_score") or 25  # fallback for mock data
        confidence = entry.get("confidence")

        # Original logic: all signals included
        original_confidences.append(confidence)

        # Simulated logic: only keep signals where RSI < new threshold
        if rsi_value < new_threshold:
            shadow_confidences.append(confidence)

    return original_confidences, shadow_confidences


def summarize_results(original, shadow):
    """Print a comparison of original vs shadow confidence performance."""

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0

    print("\n📊 SHADOW TEST RESULTS")
    print(f"• Total trades (original): {len(original)}")
    print(f"• Total trades (shadow):   {len(shadow)}")
    print(f"• Avg confidence (original): {avg(original)}")
    print(f"• Avg confidence (shadow):   {avg(shadow)}")

    improvement = avg(shadow) - avg(original)
    print(f"• Δ Confidence improvement: {improvement:+.4f}")

    # Shadow test approval rule
    if improvement > 0 and len(shadow) >= 0.5 * len(original):
        print("\n✅ Shadow test PASSED — this change would be eligible for live deployment.")
    else:
        print("\n❌ Shadow test FAILED — not enough improvement or too few trades.")


if __name__ == "__main__":
    logging.info("🧪 Running shadow test simulation...")
    ledger = load_ledger()
    if not ledger:
        logging.warning("No valid entries to test.")
        raise SystemExit(0)

    original_scores, shadow_scores = simulate_rsi_threshold(
        ledger,
        new_threshold=SUGGESTED_RSI_THRESHOLD,
    )
    summarize_results(original_scores, shadow_scores)
