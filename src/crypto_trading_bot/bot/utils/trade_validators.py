"""
Trade Validator Module

Provides schema validation for trade log entries to ensure required fields are present.
"""


def validate_trade_schema(trade_data):
    """
    Validates that a trade entry contains all required fields.

    Args:
        trade_data (dict): Dictionary representing a trade.

    Returns:
        bool: True if all required fields are present, False otherwise.
    """
    required_fields = ["timestamp", "pair", "size", "strategy", "confidence", "status"]
    for field in required_fields:
        if field not in trade_data or trade_data[field] in [None, "", "N/A"]:
            return False
    return True
