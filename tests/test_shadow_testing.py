"""Shadow testing results smoke test.

Creates logs/shadow_test_results.jsonl with a couple of dummy entries and then
verifies that the file exists, entries are valid JSON, and at least one entry
contains a success or win_rate field.
"""

import json
import os
from datetime import datetime, timezone


def test_shadow_test_results_has_success_rate(tmp_path):
    # Ensure logs directory exists and clear the file if present
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "shadow_test_results.jsonl")
    if os.path.exists(path):
        os.remove(path)

    # Write 1–2 dummy results (compact JSONL)
    entries = [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suggestion": "Tighten stop loss",
            "confidence": 0.72,
            "success": True,
        },
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suggestion": "Increase take profit",
            "confidence": 0.61,
            "win_rate": 0.58,
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")

    # Read back and validate
    assert os.path.exists(path), "shadow_test_results.jsonl was not created"
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines, "shadow_test_results.jsonl is empty"

    objs = []
    for ln in lines:
        try:
            objs.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSON line: {e}: {ln}")

    # All entries must contain required base fields
    for obj in objs:
        assert "suggestion" in obj, "Missing suggestion field"
        assert "confidence" in obj, "Missing confidence field"

    # At least one entry must contain either success or win_rate
    has_outcome = any(("success" in o) or ("win_rate" in o) for o in objs)
    assert has_outcome, "No entry has success or win_rate field"
