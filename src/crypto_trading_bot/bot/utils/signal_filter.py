"""
Signal Filter Utility
Prevents duplicate trades on the same asset in a single evaluation cycle.
"""

from typing import List, Dict


def filter_duplicate_signals(trade_signals: List[Dict]) -> List[Dict]:
    """
    Filters out duplicate trade signals for the same asset, keeping the one with the highest
    confidence.

    Args:
        trade_signals (List[Dict]): List of trade signal dictionaries.

    Returns:
        List[Dict]: Filtered list with at most one signal per asset.
    """
    unique_signals = {}

    for signal in trade_signals:
        asset = signal["asset"]
        score = signal["signal_score"]

        if asset not in unique_signals or score > unique_signals[asset]["signal_score"]:
            unique_signals[asset] = signal

    return list(unique_signals.values())
