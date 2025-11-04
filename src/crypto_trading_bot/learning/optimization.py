"""
Module for generating and exporting optimization suggestions based on trading performance reports.
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def _clamp(v: float, lo: float = 0.1, hi: float = 1.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = lo
    return max(lo, min(hi, x))


def _bounded_param_change(current: float | None, *, pct: float = 0.2, direction: int = -1) -> tuple[float, float]:
    """Return (before, after) with a bounded ±pct change.

    If current is None, assume a sane default of 1.0 before and apply direction.
    """
    try:
        before = float(current) if current is not None else 1.0
    except (TypeError, ValueError):
        before = 1.0
    delta = before * pct * (1 if direction >= 0 else -1)
    after = before + delta
    return before, after


def generate_suggestions(report: Dict) -> List[Dict]:
    """
    Generate parameter-level learning suggestions as structured records:
    {"type":"learning_suggestion","strategy":"<name>",
     "param_name":"<parameter>","param_value_before":<float>,"param_value_after":<float>,
     "confidence_before":<float>,"confidence_after":<float>,
     "reason":"<text>","timestamp":"<utc-iso>","status":"pending"}
    """
    ts = datetime.now(timezone.utc).isoformat()
    win_rate = float(report.get("win_rate", 0) or 0)
    sharpe = float(report.get("sharpe_ratio", 0) or 0)
    sortino = float(report.get("sortino_ratio", 0) or 0)
    drawdown = float(report.get("max_drawdown", 0) or 0)
    roi_pct = float(report.get("roi_percent", 0) or 0)

    out: List[Dict] = []

    # Heuristic mappings to concrete strategies/params commonly used
    # Adjust RSI upper/lower bands on poor win rate
    if win_rate < 0.4:
        # Tighten RSI entry by lowering upper or raising lower band
        before_u, after_u = _bounded_param_change(70.0, pct=0.2, direction=-1)
        conf_b, conf_a = 0.7, _clamp(0.7)  # confidence unchanged but clamped
        out.append(
            {
                "type": "learning_suggestion",
                "strategy": "SimpleRSIStrategy",
                "param_name": "rsi_upper",
                "param_value_before": round(before_u, 6),
                "param_value_after": round(after_u, 6),
                "confidence_before": conf_b,
                "confidence_after": conf_a,
                "reason": f"Low win rate detected ({win_rate:.2f}); tighten RSI upper threshold",
                "timestamp": ts,
                "status": "pending",
            }
        )
        before_l, after_l = _bounded_param_change(30.0, pct=0.2, direction=+1)
        out.append(
            {
                "type": "learning_suggestion",
                "strategy": "SimpleRSIStrategy",
                "param_name": "rsi_lower",
                "param_value_before": round(before_l, 6),
                "param_value_after": round(after_l, 6),
                "confidence_before": conf_b,
                "confidence_after": conf_a,
                "reason": f"Low win rate detected ({win_rate:.2f}); raise RSI lower threshold",
                "timestamp": ts,
                "status": "pending",
            }
        )

    # Sharpe low but Sortino acceptable -> improve exits (tighten stops)
    if sharpe < 1 and sortino > 1:
        before_sl, after_sl = _bounded_param_change(0.015, pct=0.2, direction=-1)
        conf_b, conf_a = 0.6, _clamp(0.6)
        out.append(
            {
                "type": "learning_suggestion",
                "strategy": "CompositeStrategy",
                "param_name": "stop_loss_pct",
                "param_value_before": round(before_sl, 6),
                "param_value_after": round(after_sl, 6),
                "confidence_before": conf_b,
                "confidence_after": conf_a,
                "reason": f"Sharpe low ({sharpe:.2f}) with Sortino acceptable ({sortino:.2f}); tighten SL",
                "timestamp": ts,
                "status": "pending",
            }
        )

    # High drawdown -> reduce position sizing buffer
    if abs(drawdown) > 0.2:
        before_buf, after_buf = _bounded_param_change(1.0, pct=0.2, direction=-1)
        conf_b, conf_a = 0.8, _clamp(0.8)
        out.append(
            {
                "type": "learning_suggestion",
                "strategy": "CompositeStrategy",
                "param_name": "capital_buffer",
                "param_value_before": round(before_buf, 6),
                "param_value_after": round(after_buf, 6),
                "confidence_before": conf_b,
                "confidence_after": conf_a,
                "reason": f"High max drawdown ({drawdown:.3f}); reduce buffer multiplier",
                "timestamp": ts,
                "status": "pending",
            }
        )

    # Strong ROI and Win rate -> consider loosening thresholds slightly
    if roi_pct > 50 and win_rate > 0.55:
        before_u, after_u = _bounded_param_change(70.0, pct=0.2, direction=+1)
        conf_b, conf_a = 0.85, _clamp(0.9)
        out.append(
            {
                "type": "learning_suggestion",
                "strategy": "SimpleRSIStrategy",
                "param_name": "rsi_upper",
                "param_value_before": round(before_u, 6),
                "param_value_after": round(after_u, 6),
                "confidence_before": conf_b,
                "confidence_after": conf_a,
                "reason": f"Strong ROI ({roi_pct:.2f}%) with solid win rate ({win_rate:.2f}); loosen upper",
                "timestamp": ts,
                "status": "pending",
            }
        )

    # If no specific suggestion triggered, emit a monitoring record with canonical keys
    if not out:
        conf_b = conf_a = _clamp(0.7)
        out.append(
            {
                "type": "learning_suggestion",
                "strategy": "general",
                "param_name": None,
                "param_value_before": None,
                "param_value_after": None,
                "confidence_before": conf_b,
                "confidence_after": conf_a,
                "reason": "All metrics within acceptable thresholds.",
                "timestamp": ts,
                "status": "pending",
            }
        )

    return out


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
        writer = csv.DictWriter(f, fieldnames=["category", "suggestion", "confidence", "reason"])
        writer.writeheader()
        writer.writerows(suggestions)

    with open(latest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "suggestion", "confidence", "reason"])
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

        rois = [t.get("roi", 0.0) for t in group if isinstance(t.get("roi"), (int, float))]
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
