"""
nsga3_optimizer.py

Nightly NSGA-III hyperparameter optimizer for the PPO trading stack. Evolves
multi-objective configurations using historical trade metrics retrieved from
SQLite, producing candidate PPO/twap/risk parameters for future runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from crypto_trading_bot.config import CONFIG, save_config
from crypto_trading_bot.utils.sqlite_logger import log_learning_feedback

DB_TRADES_PATH = Path("db/trades.db")
OUTPUT_DIR = Path("optimization/nsga3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PARAMS_PATH = OUTPUT_DIR / "best_params.json"


@dataclass
class TradeMetrics:
    average_roi: float
    roi_std: float
    max_drawdown: float
    sample_count: int


def load_trade_metrics(db_path: Path) -> TradeMetrics:
    if not db_path.exists():
        # Synthetic baseline when historical data is unavailable.
        return TradeMetrics(average_roi=0.01, roi_std=0.02, max_drawdown=0.06, sample_count=100)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT payload FROM trades WHERE payload IS NOT NULL").fetchall()
    finally:
        conn.close()

    rois: List[float] = []
    equity_curve: List[float] = []
    cumulative = 1.0
    for row in rows:
        try:
            record = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        roi_raw = record.get("roi")
        if roi_raw is None:
            continue
        try:
            roi = float(roi_raw)
        except (TypeError, ValueError):
            continue
        rois.append(roi)
        cumulative *= 1.0 + roi
        equity_curve.append(cumulative)

    if not rois:
        return TradeMetrics(average_roi=0.01, roi_std=0.02, max_drawdown=0.06, sample_count=0)

    rois_arr = np.asarray(rois, dtype=np.float64)
    avg_roi = float(np.mean(rois_arr))
    std_roi = float(np.std(rois_arr) + 1e-6)
    max_drawdown = 0.0
    peak = -math.inf
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            dd = (value - peak) / peak
            max_drawdown = min(max_drawdown, dd)
    max_drawdown = abs(max_drawdown)
    return TradeMetrics(
        average_roi=avg_roi,
        roi_std=std_roi,
        max_drawdown=max_drawdown,
        sample_count=len(rois_arr),
    )


def simulate_objectives(
    params: Dict[str, float],
    metrics: TradeMetrics,
) -> Tuple[float, float, float]:
    """Heuristic objective simulator based on historical trade metrics."""

    min_conf = params["min_confidence"]
    kl_threshold = params["kl_threshold"]
    twap_slices = params["twap_slices"]
    risk_buffer = params["risk_buffer"]
    drawdown_multiplier = params["drawdown_multiplier"]
    min_shadow_trades = params["min_shadow_trades"]

    base_roi = metrics.average_roi or 0.005
    base_std = max(metrics.roi_std, 1e-4)
    base_drawdown = metrics.max_drawdown or 0.05

    conf_center = 0.6
    conf_factor = 1.0 + 0.25 * (1.0 - abs(min_conf - conf_center) / 0.3)
    kl_factor = 1.0 + 0.15 * (1.0 - abs(kl_threshold - 0.03) / 0.02)
    risk_center = 0.15
    risk_factor = 1.0 + 0.2 * (1.0 - abs(risk_buffer - risk_center) / 0.1)

    roi = base_roi * conf_factor * kl_factor * risk_factor
    roi *= 1.0 + 0.05 * math.log1p(metrics.sample_count / 100.0)

    twap_penalty = 1.0 + 0.1 * abs(twap_slices - 8) / 12.0
    sharpe = (roi / base_std) / twap_penalty
    sharpe *= 1.0 + 0.1 * math.log1p(min_shadow_trades / 100.0)

    drawdown = base_drawdown
    drawdown *= (drawdown_multiplier / 12.5) ** 0.5
    drawdown *= 1.0 - 0.15 * (min_conf - 0.5)
    drawdown *= 1.0 - 0.1 * math.log1p(min_shadow_trades / 100.0)
    drawdown = max(0.01, min(drawdown, 0.2))

    return roi, sharpe, drawdown


class PPOHyperParameterProblem(ElementwiseProblem):
    def __init__(self, trade_metrics: TradeMetrics):
        xl = np.array([0.3, 0.01, 2, 0.05, 5.0, 50], dtype=float)
        xu = np.array([0.9, 0.05, 20, 0.3, 20.0, 200], dtype=float)
        super().__init__(n_var=6, n_obj=3, n_constr=2, xl=xl, xu=xu, elementwise=True)
        self.trade_metrics = trade_metrics

    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs) -> None:
        min_confidence = float(x[0])
        kl_threshold = float(x[1])
        twap_slices = int(round(float(x[2])))
        risk_buffer = float(x[3])
        drawdown_multiplier = float(x[4])
        min_shadow_trades = int(round(float(x[5])))

        params = {
            "min_confidence": min_confidence,
            "kl_threshold": kl_threshold,
            "twap_slices": max(2, min(20, twap_slices)),
            "risk_buffer": risk_buffer,
            "drawdown_multiplier": drawdown_multiplier,
            "min_shadow_trades": max(50, min(200, min_shadow_trades)),
        }

        roi, sharpe, drawdown = simulate_objectives(params, self.trade_metrics)
        out["F"] = np.array([-roi, -sharpe, drawdown], dtype=float)
        out["G"] = np.array(
            [
                50 - params["min_shadow_trades"],
                drawdown - 0.08,
            ],
            dtype=float,
        )
        out["data"] = {"params": params, "metrics": (roi, sharpe, drawdown)}


def select_best_candidate(results: Sequence[Dict]) -> Dict:
    def score(entry: Dict) -> float:
        roi, sharpe, drawdown = entry["metrics"]
        return (roi * 0.6) + (sharpe * 0.4) - drawdown * 0.8

    return max(results, key=score)


def save_best_candidate(best: Dict, path: Path) -> None:
    payload = {
        "params": best["params"],
        "metrics": {
            "average_roi": best["metrics"][0],
            "sharpe": best["metrics"][1],
            "drawdown": best["metrics"][2],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def log_best_candidate(best: Dict) -> None:
    suggestion_id = f"nsga3-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    payload = {
        "suggestion_id": suggestion_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": "nsga3_optimizer",
        "status": "completed",
        "parameters": best["params"],
        "action": None,
        "accepted": True,
        "reward": best["metrics"][0],
        "actual_roi": best["metrics"][0],
        "metrics": {
            "sharpe": best["metrics"][1],
            "drawdown": best["metrics"][2],
        },
        "model_version": str(BEST_PARAMS_PATH),
    }
    log_learning_feedback(payload)


def minimise_hyperparameters(trade_metrics: TradeMetrics, *, pop_size: int, n_gen: int, seed: int) -> Dict:
    problem = PPOHyperParameterProblem(trade_metrics)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=6)
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
        save_history=False,
    )

    candidates: List[Dict] = []
    for x, data in zip(res.X, res.opt.get("data")):
        candidates.append(data)
    if not candidates:
        for x, data in zip(res.X, res.pop.get("data")):
            if data:
                candidates.append(data)
    if not candidates:
        raise RuntimeError("NSGA-III optimisation produced no candidates.")

    return select_best_candidate(candidates)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSGA-III PPO hyperparameter optimiser.")
    parser.add_argument("--db-path", type=Path, default=DB_TRADES_PATH, help="Path to trades database.")
    parser.add_argument("--pop-size", type=int, default=50, help="NSGA-III population size.")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned run without executing.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    if getattr(args, "dry_run", False):
        print("DRY RUN: Would run NSGA-III with", vars(args))
        return 0
    metrics = load_trade_metrics(args.db_path)
    best = minimise_hyperparameters(
        metrics,
        pop_size=args.pop_size,
        n_gen=args.generations,
        seed=args.seed,
    )
    save_best_candidate(best, BEST_PARAMS_PATH)
    best_params = best.get("params", {})
    if isinstance(best_params, dict):
        CONFIG.update(best_params)
        try:
            save_config()
        except Exception as exc:  # pragma: no cover - filesystem issues
            print(f"WARNING: Failed to persist CONFIG: {exc}")
    print(f"CONFIG UPDATED: min_confidence={best_params.get('min_confidence')}")
    log_best_candidate(best)
    print(
        "NSGA-III optimisation completed | avg_roi={:.6f} sharpe={:.6f} drawdown={:.6f}".format(
            best["metrics"][0],
            best["metrics"][1],
            best["metrics"][2],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
