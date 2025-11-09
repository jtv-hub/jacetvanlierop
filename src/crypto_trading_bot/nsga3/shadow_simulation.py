# pylint: disable=duplicate-code
"""
shadow_simulation.py
---------------------------------
Phase 3 — Objective Evaluation + Shadow Simulation
--------------------------------------------------
Evaluates top Pareto individuals from NSGA-3 on *real historical data*
using the existing simulation/backtest pipeline.

Responsibilities:
- Run 500-trade shadow simulations per top individual
- Compute real ROI, max drawdown, win rate, and derived metrics
- Gate individuals by risk-adjusted ROI (rar ≥ 0.18)
- Log full results for auditing and promotion
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

from crypto_trading_bot.db.sqlite_adapter import SQLiteAdapter
from crypto_trading_bot.nsga3.integration_hook import promote_to_learning_machine
from crypto_trading_bot.nsga3.regime_utils import normalize_regime_label

LOGGER = logging.getLogger(__name__)

# --- Optional imports resolved dynamically to avoid tight coupling ---
try:
    from crypto_trading_bot.bot.simulation import run_simulation
except ImportError:
    run_simulation = None

try:
    from crypto_trading_bot.backtest import run_backtest
except ImportError:
    run_backtest = None


def _safe_run_simulation(
    params: Dict[str, Any], recent_trades: List[Dict[str, Any]], max_trades: int
) -> Dict[str, Any]:
    """
    Wrapper around run_simulation() or run_backtest() to produce consistent results.
    Falls back gracefully if either function fails.
    """
    try:
        if run_simulation:
            return run_simulation(
                strategy_params=params,
                historical_data=recent_trades,
                initial_capital=10_000,
                max_trades=max_trades,
            )
        if run_backtest:
            return run_backtest(
                config=params,
                pairs=["BTC/USDC"],
                max_trades=max_trades,
                slippage_bps=5.0,
                fees_bps=10.0,
            )
        raise RuntimeError("No simulation backend available.")
    except Exception as exc:  # pragma: no cover - defensive wrapper  # pylint: disable=broad-exception-caught
        raise RuntimeError(f"Simulation failed: {exc}") from exc


def run_shadow_tests(  # pylint: disable=too-many-locals
    top_individuals: List[Any],
    max_trades: int = 500,
    regime: str = "global",
) -> List[Dict[str, Any]]:
    """
    Run full shadow simulations for top Pareto individuals and log metrics.

    Args:
        top_individuals: List of Individual objects or dicts with 'params'
        max_trades: Number of simulated trades per individual

    Returns:
        List of result dicts with computed metrics and statuses.
    """
    os.makedirs("logs", exist_ok=True)
    label = normalize_regime_label(regime)
    shadow_log_path = f"logs/nsga3_shadow_results_{label}.jsonl"
    results: List[Dict[str, Any]] = []

    # Pull recent trades from SQLite for contextual replay data
    db = SQLiteAdapter()
    try:
        recent_trades = db.get_recent_trades(limit=max_trades * 2)
    except Exception as db_err:  # pragma: no cover - sqlite optional  # pylint: disable=broad-exception-caught
        LOGGER.exception("ShadowSim: could not load trades from SQLite: %s", db_err)
        recent_trades = []

    for ind in top_individuals:
        params = getattr(ind, "params", ind.get("params"))
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "status": "pending",
            "metrics": {},
            "regime": label,
        }

        try:
            sim_result = _safe_run_simulation(params, recent_trades, max_trades)

            # Extract core metrics safely
            total_pnl = float(
                getattr(
                    sim_result,
                    "total_pnl",
                    sim_result.get("total_pnl", 0.0),
                )
            )
            drawdown = float(
                getattr(
                    sim_result,
                    "max_drawdown",
                    sim_result.get("max_drawdown", sim_result.get("max_drawdown_pct", 0.0)),
                )
            )
            win_rate = float(getattr(sim_result, "win_rate", sim_result.get("win_rate", 0.0)))
            trade_count = int(
                getattr(
                    sim_result,
                    "trade_count",
                    sim_result.get("trade_count", max_trades),
                )
            )

            # Compute normalized ROI (PNL ÷ capital)
            capital = 10_000.0
            roi = total_pnl / capital
            rar = roi * win_rate / (drawdown + 1e-6)
            param_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()

            record["metrics"] = {
                "roi": roi,
                "drawdown": drawdown,
                "win_rate": win_rate,
                "risk_adjusted_roi": rar,
                "trade_count": trade_count,
            }
            record["status"] = "completed"
            record["param_hash"] = param_hash
            record["rar"] = rar

            # Promotion gate
            if rar >= 0.18:
                promote_to_learning_machine(
                    [{"params": params, "metrics": record["metrics"], "rar": rar}],
                    generation=0,
                    reason="shadow_pass",
                    regime=regime,
                )

        except Exception as exc:  # pragma: no cover - simulation fallback  # pylint: disable=broad-exception-caught
            LOGGER.exception("ShadowSim: simulation failed for params %s", params)
            record["status"] = "failed"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            record["param_hash"] = hashlib.sha256(
                json.dumps(params, sort_keys=True).encode("utf-8"),
            ).hexdigest()
            record["rar"] = None

        # Append to log file
        try:
            with open(shadow_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        except Exception as log_err:  # pragma: no cover - logging fallback  # pylint: disable=broad-exception-caught
            LOGGER.exception("ShadowSim: logging error: %s", log_err)

        results.append(record)

    return results


if __name__ == "__main__":
    print("[ShadowSim] Starting dry-run test …")
    dummy_individuals = [
        {
            "params": {"rsi_low": 30, "rsi_high": 80},
            "objectives": {"roi": 0.2, "drawdown": 0.08, "win_rate": 0.62},
        },
        {
            "params": {"rsi_low": 25, "rsi_high": 75},
            "objectives": {"roi": 0.1, "drawdown": 0.05, "win_rate": 0.58},
        },
    ]
    output = run_shadow_tests(dummy_individuals)
    print(json.dumps(output, indent=2))
    print("[ShadowSim] Dry-run complete.")
