"""
market_data.py

Production market data helpers. No mock fallbacks.

This module intentionally avoids importing any mock generators. If a live
price cannot be fetched, a warning is logged and None is returned.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from crypto_trading_bot.ledger.trade_ledger import system_logger
from crypto_trading_bot.utils.price_feed import get_current_price


def get_market_snapshot(trading_pair: str) -> Optional[dict]:
    """
    Fetch a minimal real-time market snapshot for a given trading pair.

    Returns a dict with timestamp, pair, and price on success; otherwise None.
    """
    price = get_current_price(trading_pair)
    if price is None:
        system_logger.warning("Live data unavailable for %s; returning None", trading_pair)
        return None
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "pair": trading_pair,
        "price": float(price),
    }
