import unittest

from crypto_trading_bot.bot.utils.historical_data_loader import load_dummy_price_data
from crypto_trading_bot.optimizer.genetic_optimizer import evaluate_fitness
from crypto_trading_bot.bot.strategies.simple_rsi_strategies import SimpleRSIStrategy
from crypto_trading_bot.bot.strategies.dual_threshold_strategies import DualThresholdStrategy


class TestGeneticOptimizer(unittest.TestCase):
    def test_evaluate_fitness_runs(self):
        prices = load_dummy_price_data()
        simple_grid = [
            {"period": 14, "lower": 40, "upper": 70},
            {"period": 21, "lower": 48, "upper": 75},
        ]
        dual_grid = [
            {},
        ]
        simple_results = evaluate_fitness(SimpleRSIStrategy, simple_grid, prices)
        dual_results = evaluate_fitness(DualThresholdStrategy, dual_grid, prices)
        # Ensure results exist and scores in [0,1]
        self.assertTrue(len(simple_results) > 0)
        self.assertTrue(len(dual_results) > 0)
        for _, s in simple_results + dual_results:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
