# src/crypto_trading_bot/main.py

"""
Main entry point for the Crypto Trading Bot.
Allows user to run a one-off trade or start the scheduler.
"""

import argparse

from crypto_trading_bot.bot.scheduler import run_scheduler
from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade


def main():
    """
    Parses command-line arguments and runs the Crypto Trading Bot in the specified mode.
    Supports 'once' mode for a single trade
    and 'schedule' mode for continuous trading at set intervals.
    """
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["once", "schedule"],
        default="once",
        help="Run mode: 'once' for single trade, 'schedule' for continuous trading",
    )
    parser.add_argument("--pair", type=str, default="BTC/USD", help="Trading pair, e.g., BTC/USD")
    parser.add_argument("--size", type=float, default=100, help="Trade size in USD")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds for scheduler")

    args = parser.parse_args()

    if args.mode == "once":
        print(f"⚡ Running one-off trade on {args.pair}\n" f"with size {args.size}...")
        result = evaluate_signals_and_trade(args.pair, args.size)
        print(f"✅ Result: {result}")
    elif args.mode == "schedule":
        run_scheduler(trading_pair=args.pair, trade_size=args.size, interval=args.interval)


if __name__ == "__main__":
    main()
