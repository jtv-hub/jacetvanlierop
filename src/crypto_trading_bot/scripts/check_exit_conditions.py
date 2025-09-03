"""
Exit Check Script

Loads open positions, fetches latest prices (live where available),
evaluates exit conditions using PositionManager, and updates the trade log.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from crypto_trading_bot.bot.trading_logic import position_manager
from crypto_trading_bot.ledger.trade_ledger import TradeLedger
from crypto_trading_bot.utils.kraken_api import get_ticker_price

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Preserve console output format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Add rotating file handler for exit checks
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        filename="logs/exit_check.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)


def get_live_prices(positions) -> dict:
    """Fetch live prices for all pairs present in open positions.

    Falls back to the first seen entry_price for a pair if live fetch fails.
    """
    prices: dict[str, float] = {}
    pairs = {p.get("pair") for p in positions if p.get("pair")}
    if not pairs:
        logger.info("No open positions — skipping price fetch.")
        return {}

    # Precompute first valid entry_price per pair to avoid repeated scans.
    fallback_prices: dict[str, float] = {}
    for pos in positions:
        pair = pos.get("pair")
        entry = pos.get("entry_price")
        if pair and pair not in fallback_prices and isinstance(entry, (int, float)) and entry > 0:
            fallback_prices[pair] = float(entry)

    for pair in pairs:
        # First, attempt to fetch from API
        try:
            raw_px = get_ticker_price(pair)
        except Exception as api_err:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to fetch live price for %s: %s", pair, api_err)
            raw_px = None

        # Then, attempt to parse/cast
        px = None
        if raw_px is not None:
            try:
                px = float(raw_px)
            except (TypeError, ValueError) as parse_err:
                logger.warning(
                    "Failed to parse live price for %s: %r (%s)",
                    pair,
                    raw_px,
                    parse_err,
                )

        if px is None or px <= 0:
            entry = fallback_prices.get(pair)
            if isinstance(entry, (int, float)) and entry > 0:
                prices[pair] = float(entry)
        else:
            prices[pair] = float(px)

    if not prices:
        logger.warning("No prices available; cannot evaluate exits.")
        return {}
    return prices


def main():
    """Load positions, build a current price map, evaluate exits, and update the ledger."""

    # Ensure positions in memory
    position_manager.load_positions_from_file()

    # Sync trade ledger state with reloaded trades
    ledger = TradeLedger(position_manager)
    ledger.reload_trades()

    # Build current prices from live Kraken feed (with fallback to entry price)
    current_prices = get_live_prices(position_manager.positions.values())

    # Optional enhancement: incorporate RSI-based exits using historical prices.

    if not current_prices:
        return

    try:
        exits = position_manager.check_exits(current_prices)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Exit evaluation failed")
        return
    for trade_id, exit_price, reason in exits:
        logger.info("Closing trade %s at %s: %s", trade_id, exit_price, reason)
        try:
            ledger.update_trade(trade_id=trade_id, exit_price=exit_price, reason=reason)
            # Force trades.log sync
            with open("logs/trades.log", "a", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except (IOError, ValueError):
            logger.exception("Failed to update trade %s", trade_id)

    if not exits:
        logger.info("No exit conditions triggered or already updated.")


if __name__ == "__main__":
    main()
