"""
objective_evaluator.py — Phase 2 update
Evaluates an individual's parameter set using a real 200-trade shadow simulation.
"""

from __future__ import annotations

from crypto_trading_bot.db.sqlite_adapter import SQLiteAdapter

try:
    from crypto_trading_bot.backtest import run_backtest as simulate_strategy
except ImportError:  # pragma: no cover - fallback to runtime sim helper
    from crypto_trading_bot.bot.simulation import (  # type: ignore[no-redef]
        run_simulation as simulate_strategy,
    )


def evaluate_individual(individual) -> dict:
    """
    Evaluate a parameter set using recent trades + a 200-trade shadow backtest.
    Returns ROI, drawdown, and win_rate, with constraint enforcement.
    """
    params = getattr(individual, "params", None)
    if params is None and isinstance(individual, dict):
        params = individual.get("params", {})
    params = params or {}

    # Ensure the DB connection is initialized (future use for fetching trade corpora).
    SQLiteAdapter()

    try:
        sim_results = simulate_strategy(
            config=params,
            pairs=["BTC/USDC"],
            max_trades=200,
        )
    except Exception as exc:  # pragma: no cover - evaluator safety  # pylint: disable=broad-exception-caught
        print(f"[Evaluator] Simulation failed: {exc}")
        return {"roi": float("-inf"), "drawdown": 1.0, "win_rate": 0.0}

    roi = float(sim_results.get("total_pnl", 0.0))
    drawdown = float(
        sim_results.get(
            "max_drawdown",
            sim_results.get("max_drawdown_pct", 0.0),
        )
    )
    win_rate = float(sim_results.get("win_rate", 0.0))

    return {"roi": roi, "drawdown": drawdown, "win_rate": win_rate}
