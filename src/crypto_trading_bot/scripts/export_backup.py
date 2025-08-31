"""
export_backup.py

Backs up all core log files to a timestamped archive directory.
"""

import os
import shutil
from datetime import datetime

LOGS_DIR = "logs"
BACKUP_ROOT = os.path.join(LOGS_DIR, "backups")

FILES_TO_BACKUP = [
    "trades.log",
    "clean_trades.log",
    "learning_ledger.jsonl",
    "confidence_summary.json",
]


def export_ledger_backup():
    """
    Creates a timestamped backup folder and copies all key logs into it.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = os.path.join(BACKUP_ROOT, timestamp)

    os.makedirs(backup_dir, exist_ok=True)
    print(f"📦 Creating backup: {backup_dir}")

    for filename in FILES_TO_BACKUP:
        src_path = os.path.join(LOGS_DIR, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(backup_dir, filename)
            shutil.copy(src_path, dst_path)
            print(f"✅ Backed up: {filename}")
        else:
            print(f"⚠️ Skipped (not found): {filename}")

    print("✅ Ledger backup complete.\n")


if __name__ == "__main__":
    export_ledger_backup()
