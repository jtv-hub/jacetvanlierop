"""Inspect and summarize learning feedback entries from logs/learning_feedback.jsonl.

Usage:
  python scripts/inspect_learning_feedback.py
  python scripts/inspect_learning_feedback.py --json

Reads UTF-8 JSONL, skips malformed lines, and extracts for each entry:
  - strategy
  - parameter (best-effort from common fields)
  - value (suggested_value/value)
  - confidence (confidence or confidence_score)
  - status/result (approved/rejected/shadow_tested/etc.)

Only standard libraries are used.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List

LOG_PATH = os.path.join("logs", "learning_feedback.jsonl")


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _first(obj: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in obj:
            return obj.get(k)
    return None


def _extract_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized fields from a raw feedback entry (best-effort)."""
    strat = _first(obj, "strategy", "strategy_name")

    # parameter name and value are stored in various shapes across code paths
    parameter = _first(obj, "parameter")
    value = _first(obj, "suggested_value", "value")

    # If nested structures exist, try to derive meaningful strings
    if not parameter and isinstance(obj.get("parameters"), dict):
        # Take first key/value from a parameters dict
        try:
            parameter = next(iter(obj["parameters"].keys()))
            if value is None:
                value = obj["parameters"].get(parameter)
        except StopIteration:
            pass

    confidence = _first(obj, "confidence", "confidence_score")
    status = _first(obj, "status", "result", "outcome")

    # Normalize types for printing
    if isinstance(confidence, (int, float)):
        confidence = float(confidence)
    elif isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            pass

    return {
        "strategy": strat or "Unknown",
        "parameter": parameter,
        "value": value,
        "confidence": confidence,
        "status": status,
        "raw": obj,
    }


def _categorize_status(status: Any) -> str:
    if not isinstance(status, str):
        return "other"
    s = status.strip().lower()
    if s in {"approved", "accept", "accepted", "applied"}:
        return "passed"
    if s in {"rejected", "declined", "deny", "denied"}:
        return "failed"
    if s in {"shadow_tested", "shadow", "pending"}:
        return "other"
    return "other"


def _load_feedback(path: str) -> Dict[str, Any]:
    malformed = 0
    entries: List[Dict[str, Any]] = []
    for ln in _read_lines(path):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            malformed += 1
            continue
        entries.append(_extract_entry(obj))
    return {"entries": entries, "malformed": malformed}


def _build_summary(entries: List[Dict[str, Any]], malformed: int) -> Dict[str, Any]:
    total = len(entries)
    strategies = [e.get("strategy") or "Unknown" for e in entries]
    strat_counts = Counter(strategies)
    unique_strats = len(strat_counts)
    top_strategy = strat_counts.most_common(1)[0][0] if strat_counts else None

    passed = sum(1 for e in entries if _categorize_status(e.get("status")) == "passed")
    failed = sum(1 for e in entries if _categorize_status(e.get("status")) == "failed")

    # Recent 3 entries for preview
    recent = entries[-3:]

    return {
        "total_suggestions": total,
        "unique_strategies": unique_strats,
        "top_strategy": top_strategy,
        "passed": passed,
        "failed": failed,
        "malformed": malformed,
        "recent": [
            {
                "strategy": r.get("strategy"),
                "parameter": r.get("parameter"),
                "value": r.get("value"),
                "confidence": r.get("confidence"),
                "status": r.get("status"),
            }
            for r in recent
        ],
    }


def _print_human(summary: Dict[str, Any]) -> None:
    print("\n📈 Learning Feedback Summary\n")
    print(f"Total Suggestions: {summary['total_suggestions']}")
    print(f"Strategies Affected: {summary['unique_strategies']}")
    if summary.get("top_strategy"):
        print(f"Top Strategy: {summary['top_strategy']}")

    if summary.get("recent"):
        print("\nRecent Suggestions:")
        for r in summary["recent"]:
            strat = r.get("strategy") or "Unknown"
            param = r.get("parameter") or "(n/a)"
            val = r.get("value")
            conf = r.get("confidence")
            stat = r.get("status") or ""
            conf_txt = f" (confidence: {conf:.2f})" if isinstance(conf, (int, float)) else ""
            arrow = "→"
            print(f"• {strat}: {param} {arrow} {val}{conf_txt} — {stat}")

    if summary.get("malformed"):
        print(f"\n⚠️ {summary['malformed']} malformed lines skipped")


def main() -> None:
    """CLI entry point: read, summarize, and print learning feedback."""
    parser = argparse.ArgumentParser(description="Inspect learning_feedback.jsonl")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON output")
    args = parser.parse_args()

    if not os.path.exists(LOG_PATH):
        print("ℹ️  logs/learning_feedback.jsonl not found.")
        return

    loaded = _load_feedback(LOG_PATH)
    entries = loaded["entries"]
    malformed = loaded["malformed"]

    summary = _build_summary(entries, malformed)
    if args.json:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        _print_human(summary)


if __name__ == "__main__":
    main()
