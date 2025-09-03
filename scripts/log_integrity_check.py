"""Log Integrity Check

Scans logs/system.log, logs/trades.log, and logs/anomalies.log for:
- Error keywords (NameError, TypeError, ValueError, KeyError, Traceback)
- Malformed JSON lines (trades.log, anomalies.log)
- Missing required trade fields (trades.log)

Usage:
    python scripts/log_integrity_check.py
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

LOG_DIR = "logs"
SYSTEM_LOG = os.path.join(LOG_DIR, "system.log")
TRADES_LOG = os.path.join(LOG_DIR, "trades.log")
ANOMALIES_LOG = os.path.join(LOG_DIR, "anomalies.log")

ERROR_KEYWORDS = [
    "NameError",
    "TypeError",
    "ValueError",
    "KeyError",
    "Traceback",
]


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def scan_error_keywords(
    path: str, lines: List[str], keywords: List[str]
) -> Tuple[Dict[str, int], List[Tuple[int, str]]]:
    counts: Dict[str, int] = {k: 0 for k in keywords}
    matches: List[Tuple[int, str]] = []
    if not lines:
        return counts, matches

    pattern = re.compile("|".join(re.escape(k) for k in keywords))
    for i, ln in enumerate(lines, start=1):
        if pattern.search(ln):
            matches.append((i, ln.rstrip("\n")))
            for k in keywords:
                if k in ln:
                    counts[k] += 1
    return counts, matches


def scan_malformed_json(lines: List[str]) -> List[Tuple[int, str]]:
    issues: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines, start=1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            json.loads(ln)
        except json.JSONDecodeError as e:
            issues.append((i, f"JSONDecodeError: {e}"))
    return issues


def check_trades_required_fields(lines: List[str], required: List[str]) -> List[Tuple[int, str, List[str]]]:
    """Return list of (line_no, trade_id, missing_fields)."""
    problems: List[Tuple[int, str, List[str]]] = []
    for i, ln in enumerate(lines, start=1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            # malformed JSON handled elsewhere
            continue
        missing: List[str] = []
        for f in required:
            if f not in obj or obj.get(f) is None:
                missing.append(f)
        if missing:
            tid = obj.get("trade_id")
            problems.append((i, str(tid) if tid is not None else "<unknown>", missing))
    return problems


def print_section(title: str):
    print(f"\n== {title} ==")


def main():
    summary_counts: Dict[str, int] = {
        SYSTEM_LOG: 0,
        TRADES_LOG: 0,
        ANOMALIES_LOG: 0,
    }

    # 1) General Error Detection
    for path in (SYSTEM_LOG, ANOMALIES_LOG):
        lines = _read_lines(path)
        counts, matches = scan_error_keywords(path, lines, ERROR_KEYWORDS)
        print_section(f"Error Scan: {path}")
        if not lines:
            print("- File not found or empty")
        else:
            for ln_no, text in matches:
                print(f"- L{ln_no}: {text}")
            # Totals per keyword
            if any(counts.values()):
                print("- Totals:")
                for k in ERROR_KEYWORDS:
                    c = counts.get(k, 0)
                    if c:
                        print(f"  * {k}: {c}")
        summary_counts[path] += sum(counts.values())

    # 2) Malformed JSON Detection (trades, anomalies)
    for path in (TRADES_LOG, ANOMALIES_LOG):
        lines = _read_lines(path)
        malformed = scan_malformed_json(lines)
        print_section(f"Malformed JSON: {path}")
        if not lines:
            print("- File not found or empty")
        else:
            for ln_no, reason in malformed[:50]:  # cap output noise
                print(f"- L{ln_no}: {reason}")
            if malformed:
                print(f"- Total malformed: {len(malformed)}")
        summary_counts[path] += len(malformed)

    # 3) Missing required fields (trades.log)
    trades_lines = _read_lines(TRADES_LOG)
    required_fields = [
        "trade_id",
        "exit_price",
        "reason",
        "roi",
        "strategy",
        "pair",
        "side",
    ]
    missing_reports = check_trades_required_fields(trades_lines, required_fields)
    print_section(f"Missing Fields: {TRADES_LOG}")
    if not trades_lines:
        print("- File not found or empty")
    else:
        for ln_no, tid, missing in missing_reports[:100]:  # cap output
            fields = ",".join(missing)
            print(f"- L{ln_no} trade_id={tid} missing=[{fields}]")
        if missing_reports:
            print(f"- Total with missing fields: {len(missing_reports)}")
    summary_counts[TRADES_LOG] += len(missing_reports)

    # 4) Summary
    print_section("Summary")
    total_issues = 0
    for path in (SYSTEM_LOG, TRADES_LOG, ANOMALIES_LOG):
        count = summary_counts.get(path, 0)
        total_issues += count
        print(f"- {path}: {count} issues")

    if total_issues == 0:
        print("\n✅ All logs passed integrity checks")
    else:
        print("\n⚠️ Logs contain issues — review recommended")


if __name__ == "__main__":
    main()
