"""
Trigger one pass of paper trading evaluation.

Usage:
  python -m src.crypto_trading_bot.scripts.trigger_paper_trading
"""

from __future__ import annotations

from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.config import CONFIG


def main() -> None:
    pairs = CONFIG.get("tradable_pairs", [])
    evaluate_signals_and_trade(tradable_pairs=pairs)


if __name__ == "__main__":
    main()
