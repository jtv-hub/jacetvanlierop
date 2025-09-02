"""
learning_feedback.py

Handles logging of learning machine feedback, including accepted/rejected
suggestions, their ROI comparisons, and forward test plans for future analysis.
"""

import json
import os
from datetime import datetime

LEARNING_LOG_PATH = "logs/learning_feedback.log"


def log_learning_feedback(
    suggestion_id,
    strategy_name,
    parameters,
    accepted,
    simulated_roi,
    actual_roi,
):
    """
    Logs the result of a learning suggestion and its performance.

    Args:
        suggestion_id (str): Unique ID of the suggestion
        strategy_name (str): Name of the strategy
        parameters (dict): The parameters being evaluated
        accepted (bool): Whether the user accepted the suggestion
        simulated_roi (float): ROI from simulated backtest of rejected suggestion
        actual_roi (float): ROI from accepted strategy in live/paper trading
    """
    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "suggestion_id": suggestion_id,
        "strategy": strategy_name,
        "parameters": parameters,
        "accepted": accepted,
        "simulated_roi": simulated_roi,
        "actual_roi": actual_roi,
        "delta": actual_roi - simulated_roi,
    }

    os.makedirs(os.path.dirname(LEARNING_LOG_PATH), exist_ok=True)

    with open(LEARNING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_forward_test_plan(
    suggestion_id,
    strategy_name,
    parameters,
    test_type,
    confidence_score,
    asset,
    timeframe,
):
    """
    Logs the planned forward test for a learning suggestion.

    Args:
        suggestion_id (str): ID of the suggestion being forward tested
        strategy_name (str): Name of the strategy
        parameters (dict): Parameters being tested
        test_type (str): 'shadow' or 'live'
        confidence_score (float): Confidence score from learning machine
        asset (str): Trading pair/asset (e.g., BTC/USD)
        timeframe (str): Timeframe being tested (e.g., 5m, 1h)
    """
    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "suggestion_id": suggestion_id,
        "strategy": strategy_name,
        "parameters": parameters,
        "test_type": test_type,
        "confidence_score": confidence_score,
        "asset": asset,
        "timeframe": timeframe,
    }

    log_path = "logs/forward_test_plans.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
