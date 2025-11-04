"""
Test script for validating the trade ledger logging functionality.
Runs a few sample trades and prints their logged output.
"""

from crypto_trading_bot.ledger.trade_ledger import TradeLedger


class _PM:
    positions = {}


def main():
    """
    Execute sample trades for BTC, ETH, and SOL to test the ledger logging.
    """
    # Run a few forced trades with different ROI outcomes
    print("🚀 Running trade ledger test...")

    ledger = TradeLedger(_PM())

    # Trade 1 - BTC (normalized to USDC pairs per ledger rules)
    trade1_id = ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=1.0,
        strategy_name="TestStrategy",
        confidence=0.95,
        entry_price=40000.0,
        regime="uptrend",
        source="unit_test",
    )
    print(trade1_id)

    # Trade 2 - ETH
    trade2_id = ledger.log_trade(
        trading_pair="ETH/USDC",
        trade_size=1.0,
        strategy_name="TestStrategy",
        confidence=0.88,
        entry_price=2500.0,
        regime="choppy",
        source="unit_test",
    )
    print(trade2_id)

    # Trade 3 - SOL
    trade3_id = ledger.log_trade(
        trading_pair="SOL/USDC",
        trade_size=1.0,
        strategy_name="TestStrategy",
        confidence=0.90,
        entry_price=100.0,
        regime="unknown",
        source="unit_test",
    )
    print(trade3_id)


if __name__ == "__main__":
    main()
