"""
Check Exit Conditions (Live)

Loads open positions, fetches latest market prices via the live Kraken
price feed, evaluates exit conditions using PositionManager, and updates
trades on exit.

Assumes the package is importable as `crypto_trading_bot` (src layout).
"""

from __future__ import annotations

import logging
import os
from typing import Dict

from crypto_trading_bot.bot.trading_logic import PositionManager as TLPositionManager
from crypto_trading_bot.ledger.trade_ledger import TradeLedger
from crypto_trading_bot.utils.price_feed import get_current_price

logger = logging.getLogger("check_exit_conditions")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_current_prices(pm: TLPositionManager) -> Dict[str, float]:
    """Fetch latest prices for all pairs in the open positions.

    Falls back to position entry_price if live fetch fails, so exits can still
    be evaluated deterministically.
    """
    prices: Dict[str, float] = {}
    pairs = {pos.get("pair") for pos in pm.positions.values() if pos.get("pair")}
    for pair in sorted(pairs):
        px = get_current_price(pair)
        if px is None or px <= 0:
            # Fallback to last known entry price for that pair (first found)
            entry = next(
                (p.get("entry_price") for p in pm.positions.values() if p.get("pair") == pair),
                None,
            )
            if isinstance(entry, (int, float)) and entry > 0:
                prices[pair] = float(entry)
                logger.warning("Using entry_price fallback for %s: %s", pair, entry)
            else:
                logger.warning("No live price or fallback for %s; skipping in checks", pair)
        else:
            prices[pair] = float(px)
            logger.debug("Fetched live price for %s: %s", pair, px)
    return prices


def main() -> None:
    """Load positions, fetch prices, evaluate exits, and update trades."""
    pm = TLPositionManager()
    pm.load_positions_from_file()

    if not pm.positions:
        print("ℹ️ No open positions found.")
        return

    # Initialize ledger bound to this position manager
    ledger = TradeLedger(pm)
    ledger.reload_trades()

    current_prices = build_current_prices(pm)
    if not current_prices:
        print("⚠️ No prices available; cannot evaluate exits.")
        return

    exits = pm.check_exits(current_prices)
    for trade_id, exit_price, reason in exits:
        print(f"🚪 Closing trade {trade_id} at {exit_price:.4f}: {reason}")
        logger.info("Closing trade %s at %.4f: %s", trade_id, exit_price, reason)
        try:
            ledger.update_trade(trade_id=trade_id, exit_price=exit_price, reason=reason)
            # Force trades.log sync
            with open("logs/trades.log", "a", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except (OSError, IOError, ValueError) as e:
            logger.error("Failed to update trade %s: %s", trade_id, e)

    if not exits:
        print("ℹ️ No exit conditions triggered.")


if __name__ == "__main__":
    main()
