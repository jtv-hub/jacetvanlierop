"""
Learning Machine for Crypto Trading Bot.

Reads closed trades from trades.log, calculates performance metrics, and can
emit learning suggestions to logs/learning_feedback.jsonl for the dashboard.

This module provides both function-level APIs (run_learning_cycle) and a small
wrapper class (LearningMachine) for compatibility with scripts that expect a
class interface with generate_report().
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Dict, List

import numpy as np

from ..bot.utils.log_rotation import get_rotating_handler
from ..risk.risk_manager import get_dynamic_buffer

logger = logging.getLogger("learning_machine")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    # Review report log (historical)
    logger.addHandler(get_rotating_handler("learning_review.log"))
    # Dedicated debug/ops log for this module as requested
    logger.addHandler(get_rotating_handler("learning_machine.log"))
    logger.propagate = False


def load_trades(log_path: str = "logs/trades.log") -> List[dict]:
    """Load closed trades with valid ROI from the trades log.

    Skips malformed lines and never raises so the learning loop is resilient.
    """
    trades: List[dict] = []
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
            status = (trade.get("status") or "").lower()
            roi = trade.get("roi")
            if status == "closed" and isinstance(roi, (int, float)):
                trades.append(trade)
    return trades


def calculate_metrics(trades: List[dict]) -> Dict[str, float | int]:
    """Calculate core ROI-based metrics from closed trades."""
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
            # convenience extras for suggestion heuristics
            "roi_percent": 0.0,
            "sortino_ratio": 0.0,
        }

    rois = np.array([float(trade["roi"]) for trade in trades], dtype=float)
    wins = int(np.sum(rois > 0))
    losses = int(np.sum(rois <= 0))
    win_rate = float(wins) / float(total_trades)

    avg_roi = float(np.mean(rois))
    cumulative_return = float(np.prod(1 + rois) - 1)

    # Sharpe ratio (risk-adjusted return)
    std = float(np.std(rois))
    sharpe_ratio = float(np.mean(rois) / std) if std > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside = rois[rois < 0]
    dd = float(np.std(downside)) if downside.size > 0 else 0.0
    sortino_ratio = float(np.mean(rois) / dd) if dd > 0 else 0.0

    # Max drawdown on equity curve of (1+roi)
    cumulative = np.cumprod(1 + rois)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0

    return {
        "total_trades": int(total_trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 6),
        "avg_roi": round(avg_roi, 6),
        "cumulative_return": round(cumulative_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 6),
        "sortino_ratio": round(sortino_ratio, 6),
        "max_drawdown": round(max_drawdown, 6),
        # convenience for heuristics/UX
        "roi_percent": round(cumulative_return * 100.0, 4),
    }


def run_learning_cycle() -> Dict[str, float | int | str]:
    """Run a single learning cycle and return metrics (with timestamp/buffer)."""
    trades = load_trades()
    metrics = calculate_metrics(trades)
    metrics["timestamp"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    metrics["capital_buffer"] = get_dynamic_buffer()
    return metrics


class LearningMachine:
    """Lightweight wrapper exposing a class interface used by some scripts."""

    def generate_report(self) -> Dict[str, float | int | str]:  # pragma: no cover - thin wrapper
        return run_learning_cycle()


def run_learning_machine(output_path: str = "logs/learning_feedback.jsonl") -> int:
    """Generate simple learning suggestions and append to JSONL file.

    Uses run_learning_cycle() metrics and the optimization.generate_suggestions
    helper to produce a compact suggestion record. Returns count written.
    """
    try:
        # Imported lazily to avoid circulars and heavy deps at import time
        from .optimization import generate_suggestions  # type: ignore
    except Exception:  # pragma: no cover - fallback path

        def generate_suggestions(_report):  # type: ignore
            return [{"suggestion": "Monitor performance", "confidence": 0.5, "reason": "fallback"}]

    # Load closed trades for diagnostics and diversity checks
    trades = load_trades()
    total_trades = len(trades)
    # Confidence stats (graceful parsing)
    valid_conf: List[float] = []
    for t in trades:
        try:
            c = float(t.get("confidence"))
            valid_conf.append(c)
        except (TypeError, ValueError):
            continue
    n_conf_not_half = sum(1 for v in valid_conf if abs(v - 0.5) > 1e-9)

    # Strategy distribution
    by_strategy: Dict[str, int] = {}
    recent_samples: List[str] = []
    for t in trades[-10:]:
        tid = t.get("trade_id")
        strat = t.get("strategy") or "Unknown"
        ts = t.get("timestamp")
        if isinstance(strat, str):
            by_strategy[strat] = by_strategy.get(strat, 0) + 1
        if isinstance(tid, str) and isinstance(ts, str):
            recent_samples.append(f"{ts} • {tid} • {strat}")

    strategies_considered = len(by_strategy)

    # Emit debug summary to both console and log
    summary_line = (
        f"[LearningMachine] Loaded {total_trades} closed trades "
        f"({len(valid_conf)} with valid confidence, {n_conf_not_half} != 0.5). "
        f"Strategies={strategies_considered}: {by_strategy}"
    )
    print(summary_line)
    logger.info(summary_line)
    if recent_samples:
        for line in recent_samples:
            logger.info("[LearningMachine] Sample: %s", line)

    # Compute report metrics and generate suggestions
    report = run_learning_cycle()
    suggestions = generate_suggestions(report) or []

    # If no suggestions, write a diagnostic entry to learning_feedback.jsonl
    if not suggestions:
        if total_trades == 0:
            reason = "No closed trades available"
        elif valid_conf and n_conf_not_half == 0:
            reason = "All trades had confidence = 0.5"
        elif strategies_considered == 0:
            reason = "No strategies detected in closed trades"
        else:
            reason = "Optimizer returned no suggestions"

        # heuristically set strategy label for diagnostics
        strat_label = (
            "Multiple"
            if strategies_considered > 1
            else (next(iter(by_strategy.keys())) if strategies_considered == 1 else "Unknown")
        )

        diag = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "type": "no_suggestions_generated",
            "strategy": strat_label,
            "status": "no_suggestions_generated",
            "reason": reason,
            "total_trades": total_trades,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(diag, separators=(",", ":")) + "\n")
        msg = (
            f"[LearningMachine] No suggestions generated — reason='{reason}', "
            f"total_trades={total_trades}, strategies={strategies_considered}"
        )
        print(msg)
        logger.info(msg)
        return 0

    # Otherwise, append generated suggestions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).isoformat()
    wrote = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for s in suggestions:
            # Normalize required fields for dashboard/report compatibility
            strategy_name = s.get("strategy") or s.get("strategy_name") or s.get("category") or "Unknown"
            confidence_before = s.get("confidence_before") or s.get("current_confidence")
            confidence_after = (
                s.get("confidence_after")
                or s.get("suggested_confidence")
                or s.get("confidence")  # fallback: suggestion confidence score
            )
            status_val = s.get("status") or "pending"

            rec = {
                "timestamp": ts,
                "type": "learning_suggestion",
                "strategy": strategy_name,
                "confidence_before": confidence_before,
                "confidence_after": confidence_after,
                "reason": s.get("reason", ""),
                "status": status_val,
            }
            # Preserve any additional fields from the generator for transparency
            for k in ("parameter", "old_value", "new_value", "suggestion", "category"):
                if k in s and k not in rec:
                    rec[k] = s[k]

            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            wrote += 1

            # Inline debug for each suggestion with full record echo
            print(f"[LearningMachine] Writing suggestion #{wrote}: {rec}")
            logger.info(
                "[LearningMachine] Suggestion #%s: strategy=%s before=%s after=%s status=%s reason=%s",
                wrote,
                rec.get("strategy"),
                rec.get("confidence_before"),
                rec.get("confidence_after"),
                rec.get("status"),
                rec.get("reason"),
            )
    logger.info("Wrote %s suggestion(s) to %s", wrote, output_path)
    print(f"[LearningMachine] Wrote {wrote} suggestion(s) to {output_path}")
    return wrote


if __name__ == "__main__":  # pragma: no cover
    # Debug mode: run a single learning cycle, print metrics, and emit suggestions
    result = run_learning_cycle()
    print("📊 Learning Machine Metrics:", result)
    try:
        n = run_learning_machine()
        print(f"✍️  Wrote {n} suggestion(s) to logs/learning_feedback.jsonl")
    except Exception as exc:  # safety for ad-hoc runs
        print(f"⚠️ Failed to write suggestions: {exc}")
