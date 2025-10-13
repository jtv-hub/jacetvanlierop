"""Seed a dummy BTC trade into the ledger for testing exit logic."""

from crypto_trading_bot.bot.strategies.simple_rsi_strategies import SimpleRSIStrategy
from crypto_trading_bot.ledger.trade_ledger import log_trade

if __name__ == "__main__":
    dummy_strategy = SimpleRSIStrategy()
    dummy_strategy.confidence = 0.85

    log_trade(
        trading_pair="BTC/USDC",
        trade_size=100,
        strategy_instance=dummy_strategy,
        strategy_name="SimpleRSIStrategy",
        entry_price=50000.0,
        confidence=0.85,
    )

    print("[Test] Dummy trade seeded successfully.")
