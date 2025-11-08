"""
Trade Schema Validator

Ensures that each trade dictionary has all required fields and correct types.
"""

from __future__ import annotations

import json
from pathlib import Path


def validate_trade_schema(trade):
    """
    Validates the structure and data types of a trade dictionary to ensure it conforms
    to the required schema used for logging and analysis.

    Args:
        trade (dict): A dictionary containing trade information.

    Returns:
        bool: True if the trade passes all validation checks.

    Raises:
        ValueError: If a required field is missing or has invalid values.
        TypeError: If a field has the wrong data type.
    """
    required_fields = {
        "trade_id": str,
        "timestamp": str,
        "pair": str,
        "size": (int, float),
        "strategy": str,
        "confidence": (int, float),
        "status": str,
        "capital_buffer": (int, float),
        "tax_method": str,
        "cost_basis": (int, float),
        "entry_price": (int, float),
        "exit_price": (int, float, type(None)),
        "realized_gain": (int, float, type(None)),
        "holding_period_days": (int, float, type(None)),
        "roi": (int, float, type(None)),
        "reason": (str, type(None)),
        "regime": str,
    }

    for field, expected_type in required_fields.items():
        if field not in trade:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(trade[field], expected_type):
            # Shorten message to satisfy line-length without changing meaning
            actual = type(trade[field])
            raise TypeError(f"Field '{field}' should be of type {expected_type}, got {actual}")

    if trade["confidence"] < 0.0 or trade["confidence"] > 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0, got {trade['confidence']}")

    if trade["size"] <= 0:
        raise ValueError(f"Trade size must be positive, got {trade['size']}")

    if trade["entry_price"] <= 0:
        raise ValueError(f"Entry price must be positive, got {trade['entry_price']}")

    return True


def validate_trades_file(path: str) -> tuple[bool, str]:
    """Validate every trade entry in a JSONL file."""

    target = Path(path)
    if not target.exists():
        return False, f"Trades log not found at {target}"

    total = 0
    errors: list[str] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_no}: JSON decode error ({exc})")
                continue
            try:
                validate_trade_schema(record)
            except (ValueError, TypeError) as exc:
                trade_id = record.get("trade_id", "<unknown>")
                errors.append(f"Line {line_no} trade_id={trade_id}: {exc}")

    if errors:
        joined = "\n".join(errors[:20])
        more = "" if len(errors) <= 20 else f"\n...and {len(errors) - 20} more issues"
        return False, f"Trade schema validation failed:\n{joined}{more}"

    return True, f"Validated {total} trades successfully from {target}"
