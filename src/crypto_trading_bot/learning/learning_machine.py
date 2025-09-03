"""
Learning Machine for Crypto Trading Bot.

This module reads executed trades from trades.log and calculates performance
metrics (ROI-based) for strategy evaluation and continuous improvement.
"""

import datetime
import json
import logging
import os

import numpy as np

from src.crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler
from src.crypto_trading_bot.risk.risk_manager import get_dynamic_buffer

logger = logging.getLogger("learning_machine")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    rotating_handler = get_rotating_handler("learning_review.log")
    logger.addHandler(rotating_handler)


def load_trades(log_path="logs/trades.log"):
    """Load trades from the log file and return as a list of dicts.

    Fix: Include closed trades with ROI (not only executed), and skip malformed lines
    gracefully so the learning cycle never crashes.
    """
    trades = []
    if not os.path.exists(log_path):
        return trades

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trade = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = trade.get("status")
            roi = trade.get("roi")
            # Only learn from trades that have realized ROI (i.e., closed)
            if status == "closed" and isinstance(roi, (int, float)):
                trades.append(trade)
    return trades


def calculate_metrics(trades):
    """Calculate metrics based on ROI from trades."""
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_roi": 0.0,
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    rois = np.array([float(trade["roi"]) for trade in trades], dtype=float)
    wins = int(np.sum(rois > 0))
    losses = int(np.sum(rois <= 0))
    win_rate = round(wins / total_trades, 3)

    avg_roi = float(np.mean(rois))
    cumulative_return = float(np.prod(1 + rois) - 1)

    # Sharpe ratio (risk-adjusted return)
    if np.std(rois) > 0:
        sharpe_ratio = float(np.mean(rois) / np.std(rois))
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumulative = np.cumprod(1 + rois)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "total_trades": int(total_trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": win_rate,
        "avg_roi": round(avg_roi, 6),
        "cumulative_return": round(cumulative_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 6),
        "max_drawdown": round(max_drawdown, 6),
    }


def run_learning_cycle():
    """Run a single learning cycle, returning metrics for the scheduler."""
    trades = load_trades()
    metrics = calculate_metrics(trades)

    # timestamp for log entries
    metrics["timestamp"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    metrics["capital_buffer"] = get_dynamic_buffer()
    return metrics


if __name__ == "__main__":
    # Debug mode: run a single learning cycle and print metrics
    result = run_learning_cycle()
    print("📊 Learning Machine Metrics:", result)
