"""
Manual Approval Script

Loads shadow test results and filters them based on configurable thresholds.
Approved suggestions are saved to approved_suggestions.jsonl.
Rejected suggestions are logged to rejected_suggestions.jsonl.
"""

import json
from pathlib import Path

# File paths
RESULTS_FILE = Path("reports/shadow_test_results.jsonl")
APPROVED_FILE = Path("reports/approved_suggestions.jsonl")
REJECTED_FILE = Path("reports/rejected_suggestions.jsonl")

# Approval thresholds
MIN_CONFIDENCE = 0.6
MIN_ROI = 0.02
MIN_WIN_RATE = 0.55


def load_shadow_results():
    """Load shadow test results from file."""
    if not RESULTS_FILE.exists():
        print("❌ No shadow test results found.")
        return []
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def approve_suggestions(results):
    """Approve or reject suggestions based on thresholds."""
    approved = []
    rejected = []

    for r in results:
        confidence = r.get("confidence", 0)
        roi = r.get("avg_roi", 0)
        win_rate = r.get("win_rate", 0)

        if confidence >= MIN_CONFIDENCE and roi >= MIN_ROI and win_rate >= MIN_WIN_RATE:
            r["status"] = "approved"
            approved.append(r)
        else:
            r["status"] = "rejected"
            rejected.append(r)

    return approved, rejected


def save_results(results, path):
    """Save the list of results to the specified file path."""
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    """Main execution function to review and save approved/rejected suggestions."""
    print("🔍 Reviewing shadow test results for approval...")
    results = load_shadow_results()
    if not results:
        return

    approved, rejected = approve_suggestions(results)
    save_results(approved, APPROVED_FILE)
    save_results(rejected, REJECTED_FILE)

    print(f"✅ {len(approved)} suggestions approved.")
    print(f"❌ {len(rejected)} suggestions rejected.")


if __name__ == "__main__":
    main()
