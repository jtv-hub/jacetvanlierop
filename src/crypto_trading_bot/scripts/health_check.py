"""
health_check.py

Performs integrity checks on trades.log and learning_ledger.jsonl.
Detects invalid JSON, missing fields, and object reference strings.
"""

import json
import os

LOG_PATHS = {"trades": "logs/trades.log", "learning": "logs/learning_ledger.jsonl"}

REQUIRED_FIELDS = {
    "trades": ["timestamp", "pair", "size", "strategy", "confidence", "status"],
    "learning": [
        "timestamp",
        "pair",
        "strategy",
        "signal",
        "confidence",
        "price",
        "volume",
        "regime",
        "outcome",  # ✅ Required for P&L tracking
    ],
}


def check_ledger(file_path, required_fields):
    """
    Validates all entries in a log file.

    Args:
        file_path (str): Path to the log file to check.
        required_fields (list): List of fields that must be present in each entry.

    Returns:
        list of tuples: Each tuple contains (line_number, error_message)
    """
    errors = []
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return errors

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            missing = [field for field in required_fields if field not in data]
            if missing:
                errors.append((i + 1, f"Missing fields: {missing}"))
            elif any(str(val).startswith("<") for val in data.values()):
                errors.append((i + 1, "Object reference found in values"))
        except json.JSONDecodeError:
            errors.append((i + 1, "Invalid JSON"))

    return errors


def run_health_check():
    """
    Runs health checks on all key log files.
    Reports missing fields, corrupted JSON, or invalid values.
    """
    print("🩺 Running ledger health check...\n")

    for name, path in LOG_PATHS.items():
        print(f"🔍 Checking {name} log: {path}")
        errors = check_ledger(path, REQUIRED_FIELDS[name])

        if not errors:
            print("✅ No issues found.\n")
        else:
            print(f"⚠️ Found {len(errors)} issue(s):")
            for line_num, message in errors:
                print(f"  Line {line_num}: {message}")
            print()


if __name__ == "__main__":
    run_health_check()
