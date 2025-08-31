"""
confidence_feedback.py

Analyzes the relationship between predicted confidence scores and actual trade outcomes
to improve future scoring and trust calibration in the learning system.
"""

import json
from collections import defaultdict


def load_trades(trade_log_path):
    """
    Load and parse the JSON-formatted trade log.

    Args:
        trade_log_path (str): Path to the trade log file.

    Returns:
        list: List of trade dictionaries.
    """
    with open(trade_log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_trades_by_confidence_bucket(trades, bucket_size=0.1):
    """
    Group trades by confidence bucket (e.g., 0.0–0.1, 0.1–0.2, ...).

    Args:
        trades (list): List of trade dicts.
        bucket_size (float): Size of each confidence bucket.

    Returns:
        dict: { bucket_start (float): [trades] }
    """
    buckets = defaultdict(list)
    for trade in trades:
        try:
            confidence = float(trade.get("confidence", 0))
        except (ValueError, TypeError):
            confidence = 0.0
        bucket = round(confidence - (confidence % bucket_size), 2)
        buckets[bucket].append(trade)
    return dict(buckets)


def analyze_feedback(trade_log_path):
    """
    Analyze confidence-to-outcome correlation.

    Args:
        trade_log_path (str): Path to trade log.

    Returns:
        list: Analysis report per confidence bucket.
    """
    trades = load_trades(trade_log_path)
    grouped = group_trades_by_confidence_bucket(trades)
    analysis = []

    for bucket, bucket_trades in sorted(grouped.items()):
        total = len(bucket_trades)
        wins = sum(1 for t in bucket_trades if t.get("status") == "executed" and t.get("roi", 0) > 0)
        avg_roi = (sum(t.get("roi", 0) for t in bucket_trades) / total) if total > 0 else 0
        win_rate = wins / total if total > 0 else 0

        analysis.append(
            {
                "confidence_bucket": bucket,
                "trade_count": total,
                "win_rate": round(win_rate, 3),
                "avg_roi": round(avg_roi, 3),
            }
        )

    return analysis


def print_confidence_analysis(analysis_report):
    """
    Display the confidence analysis nicely formatted.

    Args:
        analysis_report (list): Output from analyze_feedback().
    """
    print("Confidence Feedback Report:")
    print(f"{'Confidence':<15} {'Trades':<15} {'Win Rate':<15} {'Avg ROI':<15}")
    for r in analysis_report:
        print(f"{r['confidence_bucket']:<15} {r['trade_count']:<15} " f"{r['win_rate']:<15} {r['avg_roi']:<15}")


if __name__ == "__main__":
    TRADE_LOG_PATH = "logs/trades.log"
    final_analysis = analyze_feedback(TRADE_LOG_PATH)
    print_confidence_analysis(final_analysis)
