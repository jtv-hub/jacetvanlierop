"""
Reset Trades Script

Clears trades.log and reseeds it with dummy trades for testing.
"""

import os
import random

from crypto_trading_bot.ledger.trade_ledger import TradeLedger


def reset_trades(num_trades: int = 20):
    """Wipe trades.log and reseed with dummy trades."""
    os.makedirs("logs", exist_ok=True)

    # Clear existing trades
    with open("logs/trades.log", "w", encoding="utf-8"):
        pass

    strategies = ["RSI", "MACD", "VWAP"]

    # Create a minimal ledger instance (position manager is unused in log_trade)
    class _PM:
        positions = {}

    ledger = TradeLedger(_PM())

    for _ in range(num_trades):
        pair = random.choice(["BTC/USDC", "ETH/USDC", "XRP/USDC"])
        size = random.randint(10, 100)
        strategy = random.choice(strategies)
        confidence = round(random.uniform(0.5, 1.0), 2)
        entry_price = round(random.uniform(100, 1000), 2)
        exit_price = round(entry_price * random.uniform(0.95, 1.05), 2)

        # Log trade entry (ledger applies entry slippage and validation)
        trade_id = ledger.log_trade(
            trading_pair=pair,
            trade_size=size,
            strategy_name=strategy,
            confidence=confidence,
            entry_price=entry_price,
        )

        # Immediately close with an exit to simulate complete lifecycle
        try:
            ledger.update_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                reason="SEEDED",
            )
        except (ValueError, OSError, IOError):
            # Best-effort: skip failures during seeding
            pass

    print(f"✅ Reset complete — seeded {num_trades} dummy trades into logs/trades.log")


if __name__ == "__main__":
    reset_trades()
