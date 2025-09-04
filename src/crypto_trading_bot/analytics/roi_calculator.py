"""ROI-based running balance utilities.

Provides a reusable function to compute a running balance from a list of
trade dicts. The calculation mirrors the CLI logic used by scripts/audit_roi.py:

- Sort trades by timestamp (ISO8601)
- Skip malformed entries and trades missing status/size/roi
- Consider only CLOSED trades
- Start from a configurable starting balance ($1000 by default)
- Update balance per trade using: balance += size * roi
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List


def _to_epoch_seconds(ts: str | None) -> float:
    """Parse ISO8601 to epoch seconds, returning 0.0 on failure."""
    if not ts or not isinstance(ts, str):
        return 0.0
    try:
        dt = datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def compute_running_balance(trades: List[dict], starting_balance: float = 1000.0) -> float:
    """Compute a running balance from closed trades using size * roi increments.

    Args:
        trades: A list of trade dictionaries (JSONL rows parsed as dicts).
        starting_balance: Initial balance to use for the running total.

    Returns:
        Final balance as a float rounded to 2 decimal places (no rounding
        is applied to individual increments to preserve fidelity).
    """
    closed = []
    for t in trades:
        try:
            if (t.get("status") or "").lower() != "closed":
                continue
            size = float(t.get("size"))
            roi = float(t.get("roi"))
            ts = t.get("timestamp")
            closed.append((size, roi, _to_epoch_seconds(ts)))
        except (TypeError, ValueError):
            # Skip malformed entries
            continue

    # Sort by timestamp
    closed.sort(key=lambda x: x[2])

    bal = float(starting_balance)
    for size, roi, _ in closed:
        bal += size * roi
    return round(bal, 2)


__all__ = ["compute_running_balance"]
