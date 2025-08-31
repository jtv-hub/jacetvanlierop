"""
Genetic Optimization Engine for Strategy Parameter Tuning

This module implements a simple genetic algorithm to optimize parameters
for trading strategies using historical price data and a fitness function.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from crypto_trading_bot.bot.utils.alert import send_alert


def evaluate_fitness(
    strategy_class,
    param_grid: List[Dict],
    prices: Dict[str, List[float]],
    scoring_mode: str = "confidence",
) -> List[Tuple[Dict, float]]:
    """
    Evaluate fitness for each parameter combination of a strategy.

    Args:
        strategy_class: Strategy class to instantiate.
        param_grid (list): List of parameter dictionaries.
        prices (dict): Dictionary of price lists by asset.
        scoring_mode (str): How to evaluate the output (default: 'confidence').

    Returns:
        List of tuples: (parameter dict, score)
    """
    results = []

    # Select asset to evaluate — use BTC by default
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
            # Pass mock volume (100) to match SimpleRSIStrategy signature
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

        except (TypeError, ValueError, AttributeError) as e:
            send_alert(
                "Error evaluating parameters in genetic optimizer",
                {"params": params, "error": str(e)},
                level="WARN",
            )
            continue

    return results


def select_top_n(results: List[Tuple[Dict, float]], n: int = 5) -> List[Tuple[Dict, float]]:
    """Select top N performing parameter sets."""
    return sorted(results, key=lambda x: x[1], reverse=True)[:n]


def crossover(parent1: Dict, parent2: Dict) -> Dict:
    """Perform crossover between two parent parameter sets."""
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


def mutate(params: Dict, mutation_rate: float = 0.1) -> Dict:
    """Randomly mutate parameter values."""
    mutated = params.copy()
    for key in mutated:
        if random.random() < mutation_rate:
            if isinstance(mutated[key], int):
                mutated[key] += random.randint(-2, 2)
            elif isinstance(mutated[key], float):
                mutated[key] += random.uniform(-1.0, 1.0)
    return mutated


def genetic_optimization(
    strategy_class,
    initial_grid: List[Dict],
    prices: Dict[str, List[float]],
    generations: int = 5,
    population_size: int = 10,
) -> List[Tuple[Dict, float]]:
    """
    Run genetic optimization for a strategy.

    Args:
        strategy_class: Strategy class to optimize.
        initial_grid: Initial list of parameter combinations.
        prices: Historical price data.
        generations: Number of iterations to evolve.
        population_size: Size of the population per generation.

    Returns:
        List of best parameter sets with scores.
    """
    population = initial_grid

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}...")
        fitness_results = evaluate_fitness(strategy_class, population, prices)
        top_performers = select_top_n(fitness_results, n=population_size // 2)

        # Crossover + Mutation to form new generation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(top_performers, 2)
            child = crossover(parent1[0], parent2[0])
            child = mutate(child)
            new_population.append(child)

        population = new_population

    final_results = evaluate_fitness(strategy_class, population, prices)
    return select_top_n(final_results)
