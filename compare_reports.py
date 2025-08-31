# compare_reports.py
"""
Utility script to verify that the latest JSON and CSV learning reports
match exactly. Helps catch data export mismatches early.
"""

import os
import json
import csv

REPORTS_DIR = "reports"

def load_latest_files():
    """Find the most recent JSON and CSV report file paths.

    Returns:
        tuple: Paths to the latest JSON and CSV report files.
    """
    files = sorted(os.listdir(REPORTS_DIR))
    json_files = [f for f in files if f.endswith(".json") and "latest" not in f]
    csv_files = [f for f in files if f.endswith(".csv") and "latest" not in f]

    if not json_files or not csv_files:
        raise FileNotFoundError("No JSON/CSV report files found in reports/.")

    latest_json = os.path.join(REPORTS_DIR, json_files[-1])
    latest_csv = os.path.join(REPORTS_DIR, csv_files[-1])
    return latest_json, latest_csv

def load_json(path):
    """Load JSON data from a file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv(path):
    """Load CSV data from a file and convert to a dictionary of floats.

    Args:
        path (str): Path to the CSV file.

    Returns:
        dict: Dictionary with keys as column headers and values as floats.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) != 1:
            raise ValueError("CSV report should contain exactly one row of metrics.")
        return {k: float(v) if v else 0.0 for k, v in rows[0].items()}

def compare_reports(json_data, csv_data):
    """Compare two dictionaries of metrics for mismatches beyond tolerance.

    Args:
        json_data (dict): Metrics from JSON report.
        csv_data (dict): Metrics from CSV report.

    Returns:
        dict: Keys with mismatched values and their (json_val, csv_val) tuples.
    """
    mismatches = {}
    for key in json_data:
        json_val = float(json_data[key])
        csv_val = float(csv_data[key])
        if abs(json_val - csv_val) > 1e-8:  # tolerance for float rounding
            mismatches[key] = (json_val, csv_val)
    return mismatches

if __name__ == "__main__":
    json_path, csv_path = load_latest_files()
    print(f"🔍 Comparing:\n  JSON: {json_path}\n  CSV:  {csv_path}")

    json_metrics = load_json(json_path)
    csv_metrics = load_csv(csv_path)

    diffs = compare_reports(json_metrics, csv_metrics)
    if diffs:
        print("❌ Mismatches found:")
        for k, (j, c) in diffs.items():
            print(f"   {k}: JSON={j}, CSV={c}")
    else:
        print("✅ Reports match perfectly!")
