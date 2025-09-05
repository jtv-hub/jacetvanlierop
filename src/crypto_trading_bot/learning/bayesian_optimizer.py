"""
Bayesian Optimizer for Confidence and RSI thresholds using scikit-optimize.

If scikit-optimize is unavailable, raise a helpful ImportError.

This module evaluates objective by reweighting recent closed trades from
logs/trades.log using a confidence threshold. It returns metrics that correlate
with ROI/Sharpe, sufficient for quick parameter exploration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Tuple


def _read_jsonl(path: str) -> Iterable[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _closed_trades(path: str = "logs/trades.log") -> List[dict]:
    out: List[dict] = []
    for r in _read_jsonl(path):
        if (r.get("status") or "").lower() != "closed":
            continue
        try:
            _ = float(r.get("roi"))
            _ = float(r.get("confidence"))
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def _sharpe_like(rois: List[float]) -> float:
    if not rois:
        return 0.0
    import math

    mean = sum(rois) / len(rois)
    var = sum((x - mean) ** 2 for x in rois) / len(rois)
    sd = math.sqrt(var)
    return mean / sd if sd > 1e-12 else 0.0


def _objective_from_trades(trades: List[dict], conf_th: float, rsi_lo: float, rsi_hi: float) -> Tuple[float, dict]:
    # Filter by confidence threshold only; RSI thresholds are placeholders due to missing RSI per-trade
    selected = [t for t in trades if float(t.get("confidence", 0.0)) >= conf_th]
    if not selected:
        return 0.0, {"trades": 0, "win_rate": 0.0, "avg_roi": 0.0, "sharpe": 0.0}
    rois = [float(t["roi"]) for t in selected]
    wins = sum(1 for r in rois if r > 0)
    avg = sum(rois) / len(rois)
    sharpe = _sharpe_like(rois)
    wr = wins / len(rois)
    # Composite score to maximize (we will return negative for minimization):
    score = 0.6 * sharpe + 0.4 * avg + 0.2 * wr
    return score, {"trades": len(selected), "win_rate": wr, "avg_roi": avg, "sharpe": sharpe}


@dataclass
class BOResult:
    confidence_after: float
    rsi_lower: float
    rsi_upper: float
    score: float
    metrics: dict


def run_bayes_optimization(n_calls: int = 20) -> BOResult:
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception as exc:
        raise ImportError("scikit-optimize (skopt) not installed. Install with: pip install scikit-optimize") from exc

    trades = _closed_trades()
    if not trades:
        # Trivial fallback
        return BOResult(confidence_after=0.5, rsi_lower=30.0, rsi_upper=70.0, score=0.0, metrics={})

    def objective(x):
        conf_th, rsi_lo, rsi_hi = x
        score, _ = _objective_from_trades(trades, conf_th, rsi_lo, rsi_hi)
        # skopt minimizes; we maximize score → return negative
        return -float(score)

    space = [
        Real(0.3, 0.9, name="confidence_threshold"),
        Real(10.0, 40.0, name="rsi_oversold"),
        Real(60.0, 90.0, name="rsi_overbought"),
    ]

    res = gp_minimize(objective, space, n_calls=max(5, int(n_calls)), random_state=42)
    conf_th_opt, rsi_lo_opt, rsi_hi_opt = res.x
    best_score, metrics = _objective_from_trades(trades, conf_th_opt, rsi_lo_opt, rsi_hi_opt)
    return BOResult(
        confidence_after=float(conf_th_opt),
        rsi_lower=float(rsi_lo_opt),
        rsi_upper=float(rsi_hi_opt),
        score=float(best_score),
        metrics=metrics,
    )


def log_suggestion(
    result: BOResult, strategy: str = "SimpleRSIStrategy", output_path: str = "logs/learning_feedback.jsonl"
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    rec = {
        "timestamp": ts,
        "type": "learning_suggestion",
        "strategy": strategy,
        "confidence_before": None,
        "confidence_after": result.confidence_after,
        "suggested_param": {
            "rsi_lower": result.rsi_lower,
            "rsi_upper": result.rsi_upper,
        },
        "reason": (f"Bayesian optimization improved composite score to {result.score:.4f}"),
        "status": "pending",
        # additional metrics for reference
        "win_rate": result.metrics.get("win_rate"),
        "avg_roi": result.metrics.get("avg_roi"),
        "sharpe": result.metrics.get("sharpe"),
        "trades": result.metrics.get("trades"),
    }
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def main(n_calls: int = 20) -> None:
    res = run_bayes_optimization(n_calls=n_calls)
    log_suggestion(res)
    print(
        "[bayesian_optimizer] Best: conf>=%.3f, rsi_lower=%.1f, rsi_upper=%.1f | score=%.4f"
        % (res.confidence_after, res.rsi_lower, res.rsi_upper, res.score)
    )


if __name__ == "__main__":
    main()
