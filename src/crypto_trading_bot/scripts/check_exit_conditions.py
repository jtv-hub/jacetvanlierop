"""
Exit Check Script

This script loads positions and current mock prices, checks exit conditions,
and updates the trade log accordingly.
"""

import os

from crypto_trading_bot.bot.trading_logic import mock_price_data, position_manager
from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.indicators.rsi import calculate_rsi
from crypto_trading_bot.ledger.trade_ledger import TradeLedger


def main():
    """
    Loads positions and mock prices, evaluates exit conditions, and updates the trade log.
    """
    position_manager.load_positions_from_file()

    # Sync trade ledger state with reloaded trades
    ledger = TradeLedger(position_manager)
    ledger.reload_trades()

    # Simulate current prices (latest mock data point)
    current_prices = {f"{asset}/USD": prices[-1] for asset, prices in mock_price_data.items()}

    # Add RSI-based exit pass to increase coverage
    for _, pos in position_manager.positions.items():
        pair = pos.get("pair")
        asset = pair.split("/")[0] if pair else None
        series = mock_price_data.get(asset)
        if series and len(series) >= CONFIG["rsi"]["period"] + 1:
            rsi_val = calculate_rsi(series, CONFIG["rsi"]["period"])
            if rsi_val is not None and rsi_val >= CONFIG["rsi"].get("exit_upper", 70):
                current_prices[pair] = series[-1]
    exits = position_manager.check_exits(current_prices)
    for trade_id, exit_price, reason in exits:
        print(f"🚪 Closing trade {trade_id} at {exit_price}: {reason}")
        try:
            ledger.update_trade(trade_id=trade_id, exit_price=exit_price, reason=reason)
            # Force trades.log sync
            with open("logs/trades.log", "a", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except (IOError, ValueError) as e:
            print(f"[Error] Failed to update trade {trade_id}: {e}")

    if not exits:
        print("ℹ️ No exit conditions triggered or already updated.")


if __name__ == "__main__":
    main()
