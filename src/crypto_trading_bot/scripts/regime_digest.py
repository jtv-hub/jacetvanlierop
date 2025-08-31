#!/usr/bin/env python3
"""
regime_digest.py

Reads the learning ledger and writes one digest per market regime to logs/reports/.
If entries have no 'regime' field, they are grouped under 'unknown'.

Output files are named like:
  logs/reports/regime_digest_YYYYMMDD_HHMMSS_<regime>.txt
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LOGS_DIR = Path("logs")
LEDGER_PATH = LOGS_DIR / "learning_ledger.jsonl"
REPORTS_DIR = LOGS_DIR / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _read_ledger(path: Path) -> List[dict]:
    """Read newline-delimited JSON entries from the ledger, skipping bad lines."""
    if not path.exists():
        logging.warning("Ledger not found at %s", path)
        return []

    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entries.append(obj)
            except json.JSONDecodeError:
                # Skip a bad row but keep going.
                continue
    return entries


def _group_by_regime(entries: Iterable[dict]) -> Dict[str, List[dict]]:
    """Group entries by 'regime' (default 'unknown')."""
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        regime = (e.get("regime") or "unknown").strip() or "unknown"
        buckets[regime].append(e)
    return buckets


def _avg_conf(entries: Iterable[dict]) -> float:
    """Average of 'confidence' for the given entries, 0.0 if none."""
    vals = [float(e.get("confidence")) for e in entries if e.get("confidence") is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _signal_counts(entries: Iterable[dict]) -> Counter:
    """Count of 'signal' values."""
    c: Counter = Counter()
    for e in entries:
        sig = (e.get("signal") or "").lower().strip()
        if sig:
            c[sig] += 1
    return c


def _write_regime_report(regime: str, rows: List[dict], timestamp: str) -> Path:
    """Write a single regime digest file and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"regime_digest_{timestamp}_{regime.replace(' ', '_')}.txt"
    out_path = REPORTS_DIR / fname

    sigs = _signal_counts(rows)
    avg_conf = _avg_conf(rows)

    # Optional: show any outcome distribution if present
    outcomes = Counter((e.get("outcome") or "missing") for e in rows)

    lines = [
        "=== REGIME DIGEST ===",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Regime: {regime}",
        "",
        f"Entries: {len(rows)}",
        f"Avg confidence: {avg_conf:.4f}",
        "",
        "Signals:",
    ]
    if sigs:
        for name, cnt in sigs.items():
            lines.append(f"  - {name.upper()}: {cnt}")
    else:
        lines.append("  (none)")

    lines.extend(["", "Outcomes:"])
    if outcomes:
        for name, cnt in outcomes.items():
            lines.append(f"  - {name}: {cnt}")
    else:
        lines.append("  (none)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    """Generate regime-specific digests from the learning ledger."""
    entries = _read_ledger(LEDGER_PATH)
    if not entries:
        logging.info("No ledger entries; nothing to write.")
        return 0

    groups = _group_by_regime(entries)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    written: List[Tuple[str, Path]] = []
    for regime, rows in groups.items():
        path = _write_regime_report(regime, rows, ts)
        written.append((regime, path))

    for regime, path in written:
        logging.info("Saved regime digest for '%s': %s", regime, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
