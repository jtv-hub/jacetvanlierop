"""
Genetic Optimization Engine for Strategy Parameter Tuning (deprecated).

This module is retained as a lightweight wrapper around the NSGA-III
optimiser so existing imports do not break.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Dict, List, Tuple

from crypto_trading_bot.utils.alerts import send_alert

warnings.warn(
    "genetic_optimization.py is deprecated. Use nsga3_optimizer.py",
    DeprecationWarning,
    stacklevel=2,
)


def evaluate_fitness(
    strategy_class,
    param_grid: List[Dict],
    prices: Dict[str, List[float]],
    scoring_mode: str = "confidence",
) -> List[Tuple[Dict, float]]:
    """
    Evaluate fitness for parameter combinations.

    Kept for compatibility with legacy tests, but callers should migrate to
    NSGA-III tooling.
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
    Deprecated entry point for the legacy GA.
    """
    warnings.warn("Simple GA is deprecated. Use NSGA-III.", DeprecationWarning, stacklevel=2)
    return []


def run_nsga3_evolution(*, generations: int = 20) -> None:
    """
    Convenience wrapper that triggers the NSGA-III optimiser.
    """
    from crypto_trading_bot.optimization.nsga3_optimizer import main

    main(["--generations", str(generations)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deprecated GA wrapper for NSGA-III helper.")
    parser.add_argument("--generations", type=int, default=20, help="Number of NSGA-III generations to run.")
    args = parser.parse_args()
    run_nsga3_evolution(generations=args.generations)
