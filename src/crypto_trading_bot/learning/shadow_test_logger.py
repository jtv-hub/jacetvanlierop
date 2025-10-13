"""
shadow_test_logger.py

Logs the results of shadow tests for rejected learning suggestions, allowing
comparison between user-accepted strategies and machine-suggested alternatives.
"""

import json
import os
from datetime import datetime, timezone

SHADOW_TEST_LOG_PATH = "logs/shadow_test_results.log"


def log_shadow_test_result(
    suggestion_id,
    strategy_name,
    parameters,
    asset,
    timeframe,
    roi,
    win_rate,
    trade_count,
    test_duration_minutes,
):
    """
    Logs the performance of a shadow-tested suggestion.

    Args:
        suggestion_id (str): ID of the rejected suggestion being shadow tested
        strategy_name (str): Strategy class name
        parameters (dict): Parameters tested
        asset (str): Trading pair (e.g., BTC/USDC)
        timeframe (str): Timeframe used (e.g., 5m)
        roi (float): Total return on investment
        win_rate (float): Win rate during the test
        trade_count (int): Number of trades executed
        test_duration_minutes (int): Length of shadow test
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "suggestion_id": suggestion_id,
        "strategy": strategy_name,
        "parameters": parameters,
        "asset": asset,
        "timeframe": timeframe,
        "roi": roi,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "test_duration_minutes": test_duration_minutes,
    }

    os.makedirs(os.path.dirname(SHADOW_TEST_LOG_PATH), exist_ok=True)

    with open(SHADOW_TEST_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
