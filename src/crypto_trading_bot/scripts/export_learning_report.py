# src/crypto_trading_bot/scripts/export_learning_report.py

"""
Export Learning Analysis Reports
Saves reports to JSON/CSV and prunes history to prevent storage bloat.
"""

import csv
import json
import logging
import os
from datetime import datetime

from crypto_trading_bot.learning.learning_machine import LearningMachine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

REPORTS_DIR = "reports"
MAX_HISTORY = 100  # keep last 100 reports


def ensure_reports_dir():
    """Create reports directory if it doesn’t exist."""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)


def prune_old_reports():
    """
    Keep only the last MAX_HISTORY report files (JSON + CSV).
    Deletes oldest files first.
    """
    files = sorted(
        [os.path.join(REPORTS_DIR, f) for f in os.listdir(REPORTS_DIR)],
        key=os.path.getmtime,
    )
    if len(files) > MAX_HISTORY * 2:  # json + csv per report
        old_files = files[: len(files) - MAX_HISTORY * 2]
        for f in old_files:
            try:
                os.remove(f)
                logger.info("Pruned old report: %s", f)
            except OSError as e:
                logger.error("Error pruning %s: %s", f, e)


def save_report_json(report, filename):
    """Save learning report as JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved JSON report: %s", filename)


def save_report_csv(report, filename):
    """Save learning report as CSV."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(report.keys())
        writer.writerow(report.values())
    logger.info("Saved CSV report: %s", filename)


def export_learning_report():
    """Generate and export learning analysis reports with pruning."""
    ensure_reports_dir()
    lm = LearningMachine()
    report = lm.generate_report()

    # File names
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_latest = os.path.join(REPORTS_DIR, "learning_report_latest.json")
    csv_latest = os.path.join(REPORTS_DIR, "learning_report_latest.csv")
    json_ts = os.path.join(REPORTS_DIR, f"learning_report_{timestamp}.json")
    csv_ts = os.path.join(REPORTS_DIR, f"learning_report_{timestamp}.csv")

    # Save files
    save_report_json(report, json_latest)
    save_report_csv(report, csv_latest)
    save_report_json(report, json_ts)
    save_report_csv(report, csv_ts)

    # Prune old reports
    prune_old_reports()

    logger.info("Export complete.")


if __name__ == "__main__":
    export_learning_report()
