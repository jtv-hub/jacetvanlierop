"""
Paper trade ledger helpers.

Appends paper-trading events (opens/closes) to
`logs/paper_trades.jsonl` as JSON Lines (JSONL).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PAPER_LEDGER = Path("logs/paper_trades.jsonl")
PAPER_LEDGER.parent.mkdir(parents=True, exist_ok=True)


def _append(row: Dict[str, Any]) -> None:
    """Append a single JSON row to the paper-trades ledger."""
    with PAPER_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def log_open(
    timestamp: str,
    pair: str,
    side: str,
    size: float,
    price: float,
    strategy: str,
    note: str | None = None,
) -> None:
    """Record an open (entry) fill."""
    _append(
        {
            "type": "open",
            "timestamp": timestamp,
            "pair": pair,
            "side": side,
            "size": float(size),
            "price": float(price),
            "strategy": strategy,
            "note": note or "",
        }
    )


def log_close(
    timestamp: str,
    pair: str,
    side: str,
    size: float,
    price: float,
    strategy: str,
    reason: str,
    note: str | None = None,
) -> None:
    """Record a close (exit) fill."""
    _append(
        {
            "type": "close",
            "timestamp": timestamp,
            "pair": pair,
            "side": side,
            "size": float(size),
            "price": float(price),
            "strategy": strategy,
            "reason": reason,
            "note": note or "",
        }
    )
