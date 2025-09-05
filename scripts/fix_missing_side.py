"""Fix missing side on closed trades in logs/trades.log.

Behavior
========
- Reads UTF-8 JSONL from logs/trades.log
- Backs up original file to logs/trades_backup_before_side_fix.jsonl
- For each trade where status == "closed": if "side" is missing/None/empty,
  set it to "long" (idempotent/safe to run repeatedly)
- Skips malformed lines without crashing
- Writes compact JSONL back to logs/trades.log
- Prints a short summary of actions taken

Usage:
  python scripts/fix_missing_side.py
"""

from __future__ import annotations

import json
import os

LOG_DIR = "logs"
TRADES_PATH = os.path.join(LOG_DIR, "trades.log")
BACKUP_PATH = os.path.join(LOG_DIR, "trades_backup_before_side_fix.jsonl")


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _needs_fix_side(val) -> bool:
    """Return True if side is missing/empty or not one of {'long','short'}."""
    if val is None:
        return True
    if isinstance(val, str):
        norm = val.strip().lower()
        if norm in {"long", "short"}:
            return False
        # empty or any other value should be fixed
        return True
    # non-string types are invalid
    return True


def main() -> None:
    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to fix.")
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    original_lines = _read_lines(TRADES_PATH)
    if not original_lines:
        print("ℹ️  trades.log is empty. Nothing to fix.")
        return

    # Backup original file (overwrite each run for safety/idempotence)
    try:
        with open(BACKUP_PATH, "w", encoding="utf-8") as bak:
            for ln in original_lines:
                bak.write((ln or "") + "\n")
    except OSError as e:
        print(f"⚠️  Failed to write backup: {e}")

    fixed_count = 0
    malformed = 0
    output_lines: list[str] = []

    for ln in original_lines:
        if not ln.strip():
            # Drop blank lines in normalized output
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            malformed += 1
            continue

        if obj.get("status") == "closed" and _needs_fix_side(obj.get("side")):
            obj["side"] = "long"
            fixed_count += 1

        output_lines.append(json.dumps(obj, separators=(",", ":")))

    # Write fixed log
    try:
        with open(TRADES_PATH, "w", encoding="utf-8") as out:
            for ln in output_lines:
                out.write(ln + "\n")
    except OSError as e:
        print(f"❌ Failed to write repaired trades.log: {e}")
        return

    print(f"✅ Fixed trades: {fixed_count}")
    print(f"⚠️ Skipped malformed lines: {malformed}")
    print(f"📁 Backup saved to {BACKUP_PATH}")


if __name__ == "__main__":
    main()
