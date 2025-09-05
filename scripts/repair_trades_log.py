"""Repair missing fields in closed trades within logs/trades.log.

Rules (applied to status == "closed" only):
  - If 'capital_buffer' is missing or null, set to 0.0
  - If 'side' is missing or null, set to 'long'
  - If 'roi' is missing, set to 0.0
  - If 'reason' is missing, set to 'UNKNOWN_EXIT'

The script reads UTF-8 JSONL, skips malformed lines with a warning, backs up the
original file to logs/trades_backup.jsonl, and writes the fixed content back to
logs/trades.log in compact JSONL format.

Usage:
  python scripts/repair_trades_log.py
"""

from __future__ import annotations

import json
import os

LOG_DIR = "logs"
TRADES_PATH = os.path.join(LOG_DIR, "trades.log")
BACKUP_PATH = os.path.join(LOG_DIR, "trades_backup.jsonl")


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _is_null_or_missing(obj: dict, key: str) -> bool:
    return (key not in obj) or (obj.get(key) is None)


def main() -> None:
    """Read trades.log, repair missing fields on closed trades, and write back."""
    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to repair.")
        return

    original_lines = _read_lines(TRADES_PATH)
    if not original_lines:
        print("ℹ️  trades.log is empty. Nothing to repair.")
        return

    fixed_lines: list[str] = []
    fixed_count = 0
    malformed_count = 0

    for _, ln in enumerate(original_lines, start=1):
        line = (ln or "").strip()
        if not line:
            # Preserve blank lines as-is
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            malformed_count += 1
            # Skip malformed rows entirely
            continue

        if obj.get("status") == "closed":
            changed = False
            if _is_null_or_missing(obj, "capital_buffer"):
                obj["capital_buffer"] = 0.0
                changed = True
            if _is_null_or_missing(obj, "side"):
                obj["side"] = "long"
                changed = True
            if "roi" not in obj:
                obj["roi"] = 0.0
                changed = True
            if "reason" not in obj:
                obj["reason"] = "UNKNOWN_EXIT"
                changed = True
            if changed:
                fixed_count += 1
            fixed_lines.append(json.dumps(obj, separators=(",", ":")))
        else:
            # Non-closed rows are written as compact JSON to preserve structure
            fixed_lines.append(json.dumps(obj, separators=(",", ":")))

    # Backup original file
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(BACKUP_PATH, "w", encoding="utf-8") as bak:
            for ln in original_lines:
                bak.write((ln or "") + "\n")
    except OSError as e:
        print(f"⚠️  Failed to write backup: {e}")

    # Write repaired file
    try:
        with open(TRADES_PATH, "w", encoding="utf-8") as out:
            for ln in fixed_lines:
                out.write(ln + "\n")
    except OSError as e:
        print(f"❌ Failed to write repaired trades.log: {e}")
        return

    print(f"✅ {fixed_count} trades fixed")
    print(f"⚠️ {malformed_count} malformed lines skipped")
    print(f"📁 Backup saved to {BACKUP_PATH}")


if __name__ == "__main__":
    main()
