"""Seed a dummy BTC trade into the ledger for testing exit logic."""

from crypto_trading_bot.bot.strategies.simple_rsi_strategies import SimpleRSIStrategy
from crypto_trading_bot.ledger.trade_ledger import TradeLedger


class _PM:
    positions = {}


if __name__ == "__main__":
    dummy_strategy = SimpleRSIStrategy()
    dummy_strategy.confidence = 0.85

    ledger = TradeLedger(_PM())

    trade_id = ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=1.0,
        strategy_name="SimpleRSIStrategy",
        entry_price=50000.0,
        confidence=0.85,
        regime="test",
        source="unit_test",
    )

    # Optionally seed an immediate close to fully test lifecycle
    ledger.update_trade(trade_id, exit_price=50500.0, reason="seed_test_exit")

    print("[Test] Dummy trade seeded successfully.")
