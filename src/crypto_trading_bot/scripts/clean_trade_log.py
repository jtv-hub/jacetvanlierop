"""
clean_trade_log.py

Cleans malformed or invalid entries from the main trades.log file.
Creates a new clean_trades.log file suitable for analytics or machine learning.
"""

import json

SOURCE_FILE = "logs/trades.log"
OUTPUT_FILE = "logs/clean_trades.log"


def is_valid(entry):
    """
    Validates whether a trade entry has the correct structure.
    Filters out malformed entries, missing fields, or bad types.
    """
    return (
        isinstance(entry.get("pair"), str)
        and isinstance(entry.get("strategy"), str)
        and not entry["strategy"].startswith("<")
        and isinstance(entry.get("confidence"), (float, int))
    )


def clean_log():
    """
    Reads trades.log, filters invalid entries, and writes cleaned entries to clean_trades.log.
    """
    with open(SOURCE_FILE, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    cleaned = []
    for line in lines:
        try:
            entry = json.loads(line)
            if is_valid(entry):
                cleaned.append(entry)
        except json.JSONDecodeError:
            continue  # Skip malformed lines

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for entry in cleaned:
            json.dump(entry, outfile)
            outfile.write("\n")

    print(f"✅ Cleaned {len(cleaned)} valid entries saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_log()
