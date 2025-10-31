"""Modernized legacy test suite exercising the NSGA-III optimiser interface."""

from __future__ import annotations

import unittest

from crypto_trading_bot.optimization.nsga3_optimizer import (
    TradeMetrics,
    minimise_hyperparameters,
    simulate_objectives,
)


class TestNSGA3Optimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = TradeMetrics(
            average_roi=0.015,
            roi_std=0.02,
            max_drawdown=0.05,
            sample_count=250,
        )

    def test_simulate_objectives_returns_positive_metrics(self) -> None:
        params = {
            "min_confidence": 0.6,
            "kl_threshold": 0.03,
            "twap_slices": 8,
            "risk_buffer": 0.15,
            "drawdown_multiplier": 12.0,
            "min_shadow_trades": 100,
        }
        roi, sharpe, drawdown = simulate_objectives(params, self.metrics)
        self.assertGreater(roi, 0.0)
        self.assertGreater(sharpe, 0.0)
        self.assertGreaterEqual(drawdown, 0.0)

    def test_minimise_hyperparameters_produces_candidate(self) -> None:
        candidate = minimise_hyperparameters(
            self.metrics,
            pop_size=12,
            n_gen=1,
            seed=1234,
        )
        self.assertIn("params", candidate)
        self.assertIn("metrics", candidate)
        roi, sharpe, drawdown = candidate["metrics"]
        self.assertIsInstance(roi, float)
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(drawdown, float)


if __name__ == "__main__":
    unittest.main()
