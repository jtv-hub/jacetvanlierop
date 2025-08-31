"""
Risk Manager Module

Provides functions to manage dynamic risk buffers and position sizing
based on performance, regime, and confidence levels.
"""


def get_dynamic_buffer():
    """
    Returns a multiplier (0.1 to 1.0) to adjust trade size based on current conditions.
    In production, this should use actual data from learning machine and regime detection.
    """

    # Placeholder values – replace these later with live metrics
    recent_drawdown = 0.12  # Mock value from learning_machine
    avg_confidence = 0.35  # Mock value from recent trades
    current_regime = "unknown"  # Options: trending, chop, volatile, unknown

    # Start with default buffer
    buffer = 1.0

    # Adjust buffer based on conditions
    if recent_drawdown > 0.2:
        buffer = 0.1  # Emergency risk-off
    elif recent_drawdown > 0.1:
        buffer = 0.5

    if avg_confidence < 0.4:
        buffer = min(buffer, 0.5)

    if current_regime == "unknown" or current_regime == "chop":
        buffer = min(buffer, 0.25)

    return buffer
