"""
Compatibility layer for legacy genetic optimiser entry points.

This module now proxies directly to the NSGA-III optimiser while keeping the
lightweight fitness helpers that existing scripts rely on. New code should
import from `crypto_trading_bot.optimization.nsga3_optimizer` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from crypto_trading_bot.optimization import nsga3_optimizer
from crypto_trading_bot.utils.alerts import send_alert

__all__ = [
    "evaluate_fitness",
    "select_top_n",
    "genetic_optimization",
    "run_nsga3_evolution",
]


def evaluate_fitness(
    strategy_class,
    param_grid: List[Dict],
    prices: Dict[str, List[float]],
    scoring_mode: str = "confidence",
) -> List[Tuple[Dict, float]]:
    """
    Evaluate fitness for parameter combinations.

    Retained for backwards compatibility so tooling that relied on the simple
    GA helper keeps working while the optimiser stack transitions to NSGA-III.
    """
    results: List[Tuple[Dict, float]] = []
    asset_prices = prices.get("BTC", [])
    if not asset_prices or len(asset_prices) < 20:
        send_alert(
            "Insufficient price data for asset BTC in genetic optimizer",
            {"asset": "BTC", "price_length": len(asset_prices)},
            level="ERROR",
        )
        return results

    for params in param_grid:
        try:
            strategy = strategy_class(**params)
            signal_output = strategy.generate_signal(asset_prices, volume=100)
            if not signal_output:
                continue
            if scoring_mode == "confidence":
                score = signal_output.get("confidence", 0.0)
            elif scoring_mode == "signal_score":
                score = signal_output.get("signal_score", 0.0)
            else:
                score = 0.0
            results.append((params, score))
        except (TypeError, ValueError, AttributeError) as exc:
            send_alert(
                "Error evaluating parameters in genetic optimizer",
                {"params": params, "error": str(exc)},
                level="WARN",
            )
    return results


def select_top_n(results: List[Tuple[Dict, float]], n: int = 5) -> List[Tuple[Dict, float]]:
    """Select top N performing parameter sets."""
    return sorted(results, key=lambda x: x[1], reverse=True)[:n]


def genetic_optimization(
    strategy_class,
    initial_grid: List[Dict],
    prices: Dict[str, List[float]],
    generations: int = 5,
    population_size: int = 10,
) -> List[Tuple[Dict, float]]:
    """
    Placeholder for the deprecated simple GA interface.

    The old genetic optimiser is no longer supported; callers should migrate to
    NSGA-III routines. We return an empty list to signal no optimisation.
    """
    _ = (strategy_class, initial_grid, prices, generations, population_size)
    return []


def run_nsga3_evolution(
    *,
    generations: int = 20,
    pop_size: int | None = None,
    seed: int | None = None,
    db_path: Path | None = None,
) -> int:
    """
    Convenience wrapper that triggers the NSGA-III optimiser.
    """
    argv: List[str] = ["--generations", str(generations)]
    if pop_size is not None:
        argv.extend(["--pop-size", str(pop_size)])
    if seed is not None:
        argv.extend(["--seed", str(seed)])
    if db_path is not None:
        argv.extend(["--db-path", str(db_path)])
    return nsga3_optimizer.main(argv)
