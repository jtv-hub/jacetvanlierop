"""
Real-time price feed utilities for external market data providers.

This module now delegates live price lookups to Kraken via
`crypto_trading_bot.utils.kraken_api.get_ticker_price` and keeps the
same public functions so existing imports continue to work.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

from crypto_trading_bot.utils.kraken_api import (
    PAIR_MAP as _PAIR_MAP,
)
from crypto_trading_bot.utils.kraken_api import (
    get_ticker_price,
)

logger = logging.getLogger(__name__)

# Simple in-memory cache to reduce rate-limit pressure.
_CACHE_TTL_SECONDS = 2.0
_cache: Dict[str, Tuple[float, float]] = {}  # key: "BASE/QUOTE" -> (ts, price)


def _now() -> float:
    return time.time()


def _kraken_to_app_pair(pair: str) -> str:
    """Convert Kraken altname like 'XBTUSD' to app format 'BTC/USD'.

    Uses inverse mapping of kraken_api.PAIR_MAP when available; otherwise,
    applies a lightweight heuristic (assumes USD quote).
    """
    inv = {v: k for k, v in _PAIR_MAP.items()}
    up = pair.upper()
    if up in inv:
        return inv[up]
    if up.endswith("USD"):
        base = up[:-3]
        if base == "XBT":
            base = "BTC"
        return f"{base}/USD"
    # Fallback: return as-is if we can't confidently translate
    return up


def _get_with_cache(app_pair: str) -> float:
    ts_now = _now()
    cached = _cache.get(app_pair)
    if cached and (ts_now - cached[0]) <= _CACHE_TTL_SECONDS:
        logger.debug("price_feed cache hit for %s: %s", app_pair, cached[1])
        return cached[1]

    price = get_ticker_price(app_pair)
    _cache[app_pair] = (ts_now, price)
    logger.debug("price_feed fetched %s -> %s", app_pair, price)
    return price


def get_kraken_price(pair: str = "XBTUSD") -> float:
    """
    Backward-compatible API: accepts a Kraken altname like 'XBTUSD' and returns
    the latest price by delegating to get_ticker_price() using app-format pairs.
    """
    app_pair = _kraken_to_app_pair(pair)
    return _get_with_cache(app_pair)


def get_current_price(pair: str = "BTC/USD") -> float | None:
    """
    Convenience wrapper that accepts a human-friendly pair like "BTC/USD"
    and returns the latest price via Kraken public API.

    Returns None if fetching fails so callers can skip gracefully.
    """
    if not isinstance(pair, str) or "/" not in pair:
        logger.warning("Invalid pair string: %r", pair)
        return None
    try:
        price = _get_with_cache(pair.upper())
        logger.info("[FEED] fetched price for %s: %s", pair, price)
        return price
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to fetch real price for %s: %s", pair, exc)
        return None
