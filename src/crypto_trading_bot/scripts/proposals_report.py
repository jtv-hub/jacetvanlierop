"""
Proposals report.

Summarizes entries from logs/learning_proposals.jsonl, showing approval
rates, average delta confidence, and the latest decisions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROPOSALS_LOG = Path("logs/learning_proposals.jsonl")
REPORTS_DIR = Path("logs/reports")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file or return an empty list."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_report(records: List[Dict[str, Any]]) -> str:
    """Build a human-readable proposals report from raw records."""
    if not records:
        return "=== PROPOSALS REPORT ===\n(no proposals found)\n"

    approvals = [r for r in records if r.get("decision", {}).get("approved")]
    rejects = [r for r in records if not r.get("decision", {}).get("approved")]

    deltas: List[float] = []
    for r in records:
        d = r.get("decision", {}).get("metrics", {}).get("delta_conf")
        if isinstance(d, (int, float)):
            deltas.append(float(d))

    lines: List[str] = []
    lines.append("=== PROPOSALS REPORT ===")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Total proposals: {len(records)}")
    lines.append(f"Approved: {len(approvals)} | Rejected: {len(rejects)}")
    if deltas:
        lines.append(f"Avg Δconfidence across proposals: {round(mean(deltas), 6)}")
    lines.append("")

    lines.append("Last 5 proposals:")
    for r in records[-5:]:
        decision = r.get("decision", {})
        status = "APPROVED" if decision.get("approved") else "REJECTED"
        metrics = decision.get("metrics", {})
        delta = metrics.get("delta_conf")
        prop = r.get("proposal")
        ts = r.get("timestamp", "?")
        lines.append(f"  - {ts} | {status} | Δconf={delta} | proposal={prop}")

    return "\n".join(lines)


def main() -> None:
    """Generate the proposals report, print it, and save under logs/reports/."""
    rows = _read_jsonl(PROPOSALS_LOG)
    text = build_report(rows)
    print("\n" + text + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"proposals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info("Saved proposals report: %s", out)


if __name__ == "__main__":
    main()
