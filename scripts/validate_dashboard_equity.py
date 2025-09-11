#!/usr/bin/env python3
"""Validate dashboard equity against audit_roi computed balance."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_LOG = PROJECT_ROOT / "logs" / "system.log"
DASHBOARD_CANDIDATES = [
    PROJECT_ROOT / "logs" / "dashboard_metrics.json",
    PROJECT_ROOT / "logs" / "reports" / "dashboard_metrics.json",
    PROJECT_ROOT / "logs" / "reports" / "dashboard_metrics.jsonl",
]
INITIAL_BALANCE = 1000.0

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts import audit_roi
except Exception as exc:  # pragma: no cover - defensive
    print(f"⚠️ Failed to import audit_roi: {exc}", file=sys.stderr)
    sys.exit(2)


def _read_dashboard_balance() -> float | None:
    for candidate in DASHBOARD_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".jsonl":
                with candidate.open("r", encoding="utf-8") as handle:
                    last_line = None
                    for line in handle:
                        line = line.strip()
                        if line:
                            last_line = line
                    if last_line:
                        data = json.loads(last_line)
                    else:
                        continue
            else:
                data = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

        for key in ("final_balance", "balance", "equity", "portfolio_value"):
            value = data.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


def _append_system_log(message: str) -> None:
    SYSTEM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SYSTEM_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{Path(__file__).name} {message} at {datetime.now(timezone.utc).isoformat()}]\n")


def main() -> int:
    dashboard_balance = _read_dashboard_balance()
    if dashboard_balance is None:
        msg = "dashboard balance unavailable"
        _append_system_log(f"{msg} @ validate_dashboard_equity")
        print(f"⚠️ {msg}")
        return 1

    audit_summary = audit_roi.run_audit(initial_balance=INITIAL_BALANCE)
    audit_balance = float(audit_summary.get("final_balance", INITIAL_BALANCE))

    diff = abs(dashboard_balance - audit_balance)
    if diff > 1.0:
        warning = (
            f"WARNING Dashboard vs Audit balance mismatch: dashboard=${dashboard_balance:.2f} "
            f"audit=${audit_balance:.2f} diff=${diff:.2f}"
        )
        _append_system_log(f"{warning} @ validate_dashboard_equity")
        print(f"⚠️ {warning}")
        return 3

    print("✅ Dashboard ROI validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
