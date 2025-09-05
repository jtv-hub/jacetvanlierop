"""Trading loop script.

Runs the main evaluation loop using the core trading logic module. It relies on
centralized configuration for tradable pairs and live pricing utilities defined
within the package. This script intentionally keeps orchestration minimal.
"""

import logging

from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.context.trading_context import TradingContext

# Optional: set DEBUG_MODE to True for more logs
DEBUG_MODE = True

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)


def main():
    """
    Main function to run the trading loop with live data.
    """
    logging.info("🟢 Starting trading loop...")
    # Touch context so it is initialized for modules that rely on it,
    # but the trading logic manages and updates context internally.
    _ = TradingContext()
    evaluate_signals_and_trade()
    logging.info("✅ Trade evaluation completed.")


if __name__ == "__main__":
    main()
