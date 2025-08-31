"""
Confidence analytics utilities.

- record_to_learning_machine(trade_metadata): append a JSONL entry to learning ledger.
- update_confidence_summary(strategy_name, confidence_score): rolling average update,
  schema-safe (handles legacy 'avg_confidence' vs 'average_confidence').
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

# Paths
LOGS_DIR = Path("logs")
LEARNING_LEDGER = LOGS_DIR / "learning_ledger.jsonl"
CONFIDENCE_SUMMARY = LOGS_DIR / "confidence_summary.json"

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------- IO helpers ----------


def _ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


# ---------- Schema normalization ----------


def _normalize_strategy_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Make sure a single strategy entry has the latest schema."""
    # Migrate legacy key
    if "avg_confidence" in entry and "average_confidence" not in entry:
        entry["average_confidence"] = entry.pop("avg_confidence")

    # Defaults
    if "average_confidence" not in entry:
        entry["average_confidence"] = 0.0
    if "trade_count" not in entry:
        entry["trade_count"] = 0
    if "regime_confidence" not in entry or not isinstance(
        entry["regime_confidence"], dict
    ):
        entry["regime_confidence"] = {}

    return entry


def _normalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize entire confidence_summary dict to the latest schema."""
    for strat, stats in list(summary.items()):
        if isinstance(stats, dict):
            summary[strat] = _normalize_strategy_entry(stats)
        else:
            # If someone stored a non-dict by accident, reset to defaults
            summary[strat] = _normalize_strategy_entry({})
    return summary


# ---------- Public API ----------


def record_to_learning_machine(trade_metadata: Dict[str, Any]) -> None:
    """
    Append a learning record to the JSONL ledger. Missing optional fields are filled.

    Expected keys:
        timestamp (str), pair (str), strategy (str), signal (str),
        confidence (float), price (number), volume (number),
        regime (str, optional), outcome (str, optional)
    """
    _ensure_logs_dir()

    # Fill optional fields if missing
    trade_metadata = dict(trade_metadata)  # shallow copy
    trade_metadata.setdefault("regime", "unknown")
    trade_metadata.setdefault("outcome", "unknown")

    # Write line
    with LEARNING_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trade_metadata) + "\n")

    logger.info("📈 Learning machine recorded: %s", trade_metadata)


def update_confidence_summary(strategy_name: str, confidence_score: float) -> None:
    """
    Update rolling average for a strategy in confidence_summary.json.

    Args:
        strategy_name: e.g., "SimpleRSIStrategy"
        confidence_score: latest confidence value to fold into the average
    """
    _ensure_logs_dir()

    # Load & normalize
    summary = _safe_read_json(CONFIDENCE_SUMMARY)
    summary = _normalize_summary(summary)

    # Current stats for this strategy
    current = summary.get(strategy_name, {})
    current = _normalize_strategy_entry(current)

    prev_avg = float(current.get("average_confidence", 0.0))
    prev_count = int(current.get("trade_count", 0))

    # Rolling average update
    new_count = prev_count + 1
    new_avg = (prev_avg * prev_count + float(confidence_score)) / new_count

    current["average_confidence"] = round(new_avg, 6)
    current["trade_count"] = new_count

    # Keep regime_confidence dict (updated elsewhere)
    if "regime_confidence" not in current or not isinstance(
        current["regime_confidence"], dict
    ):
        current["regime_confidence"] = {}

    summary[strategy_name] = current

    _atomic_write_json(CONFIDENCE_SUMMARY, summary)
    logger.info("✅ Updated confidence_summary.json for %s", strategy_name)
