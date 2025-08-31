"""
Test script for the genetic optimization engine.
Evaluates strategies using sample historical data and prints top parameters.
"""

import os

from crypto_trading_bot.bot.strategies.dual_threshold_strategies import (
    DualThresholdStrategy,
)
from crypto_trading_bot.bot.strategies.simple_rsi_strategies import SimpleRSIStrategy
from crypto_trading_bot.bot.utils.historical_data_loader import load_dummy_price_data
from crypto_trading_bot.optimizer.genetic_optimizer import evaluate_fitness

# Load sample price data
CSV_PATH = os.path.join("data", "BTCUSD_1h_sample.csv")
price_data = load_dummy_price_data()

# Define parameter grids
simple_rsi_grid = [{"period": p, "lower": lower, "upper": u} for p in [7, 14] for lower in [25, 30] for u in [70, 75]]

dual_threshold_grid = [{"lower": lower, "upper": u} for lower in [30] for u in [70]]

# Evaluate strategies
simple_results = evaluate_fitness(SimpleRSIStrategy, simple_rsi_grid, price_data, scoring_mode="confidence")
dual_results = evaluate_fitness(DualThresholdStrategy, dual_threshold_grid, price_data, scoring_mode="confidence")

# Sort results
simple_results.sort(key=lambda x: x[1], reverse=True)
dual_results.sort(key=lambda x: x[1], reverse=True)

# Print top results
print("\nTop 3 Simple RSI configurations:")
for params, score in simple_results[:3]:
    print(f"Params: {params}, Score: {score:.4f}")

print("\nTop 3 Dual Threshold configurations:")
for params, score in dual_results[:3]:
    print(f"Params: {params}, Score: {score:.4f}")
