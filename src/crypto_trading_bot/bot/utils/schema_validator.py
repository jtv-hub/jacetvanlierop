"""
Trade Schema Validator

Ensures that each trade dictionary has all required fields and correct types.
"""


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
