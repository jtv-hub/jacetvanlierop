"""
CLI entry to run the Bayesian optimizer for confidence and RSI tuning.

Usage:
  python -m src.crypto_trading_bot.scripts.run_bayesian_optimization
  or
  python src/crypto_trading_bot/scripts/run_bayesian_optimization.py
"""

from __future__ import annotations

import argparse
import sys

from crypto_trading_bot.learning.bayesian_optimizer import main as run_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Bayesian optimization for confidence/RSI")
    p.add_argument("--calls", type=int, default=20, help="Number of Bayesian optimization calls")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_main(n_calls=args.calls)


if __name__ == "__main__":
    main(sys.argv[1:])
