"""Inspect shadow testing results from logs/shadow_test_results.jsonl.

Usage:
  python scripts/inspect_shadow_results.py
  python scripts/inspect_shadow_results.py --json

Reads UTF-8 JSONL, skips malformed lines, and extracts per-entry fields:
  - strategy or strategy_name
  - result/status/outcome (best-effort)
  - win_rate or success_rate (as float if possible)
  - trades_tested or sample_size

Summarizes:
  - total entries
  - unique strategy count
  - passed (success rate >= 0.6) vs failed
  - recent 3 entries

Only standard libraries are used.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List

LOG_PATH = os.path.join("logs", "shadow_test_results.jsonl")


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


def _to_float(val: Any) -> float | None:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _extract_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
    if obj.get("type") == "strategy_confidence_summary":
        return {"skip": True, "raw": obj}
    strategy = _first(obj, "strategy", "strategy_name") or "Unknown"
    status = _first(obj, "result", "status", "outcome")
    success = _to_float(_first(obj, "win_rate", "success_rate"))
    trades_tested = _first(obj, "trades_tested", "sample_size")
    try:
        trades_tested = int(trades_tested) if trades_tested is not None else None
    except (TypeError, ValueError):
        trades_tested = None

    return {
        "strategy": strategy,
        "status": status,
        "success_rate": success,
        "trades_tested": trades_tested,
        "raw": obj,
    }


def _load_results(path: str) -> Dict[str, Any]:
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
        entry = _extract_entry(obj)
        if entry.get("skip"):
            continue
        entries.append(entry)
    return {"entries": entries, "malformed": malformed}


def _build_summary(entries: List[Dict[str, Any]], malformed: int) -> Dict[str, Any]:
    total = len(entries)
    strategies = [e.get("strategy") or "Unknown" for e in entries]
    unique_strats = len(Counter(strategies))

    def _passed(e: Dict[str, Any]) -> bool:
        sr = e.get("success_rate")
        return isinstance(sr, float) and sr >= 0.6

    passed = sum(1 for e in entries if _passed(e))
    failed = total - passed

    recent = entries[-3:]

    return {
        "total": total,
        "unique_strategies": unique_strats,
        "passed": passed,
        "failed": failed,
        "malformed": malformed,
        "recent": [
            {
                "strategy": r.get("strategy"),
                "status": r.get("status"),
                "success_rate": r.get("success_rate"),
                "trades_tested": r.get("trades_tested"),
            }
            for r in recent
        ],
    }


def _print_human(summary: Dict[str, Any]) -> None:
    print("\n🧪 Shadow Test Summary\n")
    print(f"Total suggestions tested: {summary['total']}")
    print(f"Unique strategies: {summary['unique_strategies']}")
    print(f"Passed (>=60%): {summary['passed']}")
    print(f"Failed (<60%): {summary['failed']}")

    if summary.get("recent"):
        print("\nRecent Results:")
        for r in summary["recent"]:
            strat = r.get("strategy") or "Unknown"
            status = r.get("status") or ""
            sr = r.get("success_rate")
            sr_txt = f"{sr:.2f}" if isinstance(sr, float) else "n/a"
            n = r.get("trades_tested")
            n_txt = str(n) if isinstance(n, int) else "n/a"
            print(f"• {strat}: success_rate={sr_txt}, trades={n_txt} — {status}")

    if summary.get("malformed"):
        print(f"\n⚠️ {summary['malformed']} malformed lines skipped")


def main() -> None:
    """CLI entry point: read, summarize, and print shadow test results."""
    parser = argparse.ArgumentParser(description="Inspect shadow test results JSONL")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON output")
    args = parser.parse_args()

    if not os.path.exists(LOG_PATH):
        print("ℹ️  logs/shadow_test_results.jsonl not found.")
        return

    loaded = _load_results(LOG_PATH)
    summary = _build_summary(loaded["entries"], loaded["malformed"])

    if args.json:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        _print_human(summary)


if __name__ == "__main__":
    main()
