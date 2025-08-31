"""
Module for generating and exporting optimization suggestions based on trading performance reports.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from statistics import mean, stdev

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def generate_suggestions(report: Dict) -> List[Dict]:
    """
    Generate optimization suggestions based on learning report metrics.
    Each suggestion contains: category, suggestion, confidence score, reason.
    """
    suggestions = []

    win_rate = report.get("win_rate", 0)
    sharpe = report.get("sharpe_ratio", 0)
    sortino = report.get("sortino_ratio", 0)
    drawdown = report.get("max_drawdown", 0)
    roi = report.get("roi_percent", 0)

    # --- Heuristic 1: Win Rate Low ---
    if win_rate < 0.4:
        suggestions.append(
            {
                "category": "entry_filter",
                "suggestion": "Tighten entry signal thresholds to reduce false positives.",
                "confidence": 0.7,
                "reason": f"Low win rate detected ({win_rate:.2f}).",
            }
        )

    # --- Heuristic 2: Sharpe Low but Sortino OK ---
    if sharpe < 1 and sortino > 1:
        suggestions.append(
            {
                "category": "risk",
                "suggestion": "Improve risk-adjusted returns by optimizing stop losses or trade exits.",
                "confidence": 0.6,
                "reason": (
                    f"Sharpe low ({sharpe:.2f}) while Sortino acceptable "
                    f"({sortino:.2f})."
                ),
            }
        )

    # --- Heuristic 3: High Drawdown ---
    if drawdown > 1000:
        suggestions.append(
            {
                "category": "risk",
                "suggestion": (
                    "Introduce stricter drawdown limits or reduce position sizing "
                    "in volatile regimes."
                ),
                "confidence": 0.8,
                "reason": f"Max drawdown high ({drawdown:.1f}).",
            }
        )

    # --- Heuristic 4: Strong ROI + Good Win Rate ---
    if roi > 50 and win_rate > 0.55:
        suggestions.append(
            {
                "category": "capital_allocation",
                "suggestion": (
                    "Consider increasing capital allocation for strategies "
                    "in trending regimes."
                ),
                "confidence": 0.85,
                "reason": (
                    f"Strong ROI ({roi:.2f}%) with solid win rate " f"({win_rate:.2f})."
                ),
            }
        )

    # Default if no suggestions triggered
    if not suggestions:
        suggestions.append(
            {
                "category": "general",
                "suggestion": "No immediate changes required. Continue monitoring performance.",
                "confidence": 0.5,
                "reason": "All metrics within acceptable thresholds.",
            }
        )

    return suggestions


def export_suggestions(suggestions: List[Dict]) -> None:
    """
    Export optimization suggestions to JSON and CSV in /reports.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    json_path = REPORTS_DIR / f"suggestions_{timestamp}.json"
    csv_path = REPORTS_DIR / f"suggestions_{timestamp}.csv"

    latest_json = REPORTS_DIR / "suggestions_latest.json"
    latest_csv = REPORTS_DIR / "suggestions_latest.csv"

    # JSON Export
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2)

    # CSV Export
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "suggestion", "confidence", "reason"]
        )
        writer.writeheader()
        writer.writerows(suggestions)

    with open(latest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "suggestion", "confidence", "reason"]
        )
        writer.writeheader()
        writer.writerows(suggestions)

    print(f"✅ Suggestions exported: {json_path}, {csv_path}")


# --- Step 4.6.5.1: Detect Outlier High-Performing Parameter Sets ---
def detect_outliers(min_trades=25, top_n=3):
    """Identify high-performing parameter sets from recent trade logs."""

    trade_log_path = "logs/trades.log"
    if not Path(trade_log_path).exists():
        print("[Optimization] No trades.log file found.")
        return []

    with open(trade_log_path, "r", encoding="utf-8") as f:
        trades = [json.loads(line) for line in f if line.strip()]

    strat_groups = defaultdict(list)

    for trade in trades:
        if trade.get("status") != "closed":
            continue
        key = f"{trade['strategy']}::{trade.get('params', {})}"
        strat_groups[key].append(trade)

    scored_configs = []
    for key, group in strat_groups.items():
        if len(group) < min_trades:
            continue

        rois = [
            t.get("roi", 0.0) for t in group if isinstance(t.get("roi"), (int, float))
        ]
        win_rate = sum(1 for r in rois if r > 0) / len(rois)
        avg_roi = mean(rois)
        sharpe = avg_roi / (stdev(rois) or 1e-6)

        score = round(0.5 * win_rate + 0.4 * avg_roi + 0.1 * sharpe, 4)

        scored_configs.append(
            {
                "strategy_config": key,
                "score": score,
                "avg_roi": round(avg_roi, 4),
                "win_rate": round(win_rate, 4),
                "sharpe": round(sharpe, 4),
                "count": len(group),
            }
        )

    top_configs = sorted(scored_configs, key=lambda x: x["score"], reverse=True)[:top_n]
    print(f"[Optimization] Top {len(top_configs)} strategy configs:")
    for config in top_configs:
        print(config)

    return top_configs
