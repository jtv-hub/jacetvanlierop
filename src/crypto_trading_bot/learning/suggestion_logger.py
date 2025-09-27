"""
Suggestion Logger
Appends optimization suggestions from the learning review step
into a JSONL file for historical tracking and future evaluation.
"""

import json
import os
from datetime import datetime, timezone

# Explicit file path inside learning/
SUGGESTIONS_FILE = os.path.join("learning", "suggestions.jsonl")

# Make sure directory exists
os.makedirs(os.path.dirname(SUGGESTIONS_FILE), exist_ok=True)


def log_suggestion(strategy_name, param_change, rationale, confidence, status="pending"):
    """
    Append a learning suggestion to suggestions.jsonl
    """
    suggestion_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy_name": strategy_name,
        "param_change": param_change,
        "rationale": rationale,
        "confidence": confidence,
        "status": status,
    }

    try:
        with open(SUGGESTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(suggestion_entry) + "\n")
            f.flush()

        print(f"[suggestion_logger] ✅ Wrote suggestion to {SUGGESTIONS_FILE}")
    except OSError as e:
        print(f"[suggestion_logger] ❌ Failed to write suggestion: {e}")

    return suggestion_entry
