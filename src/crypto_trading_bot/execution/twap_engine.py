"""
twap_engine.py

Time-weighted execution engine with "sweet spot" logic tailored for
maker-only limit orders. Uses order book depth and recent volatility
heuristics to determine slicing and fallback behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np

from crypto_trading_bot.config import (
    IS_PAPER_MODE,
    MAX_SLICES,
    MAX_SPREAD_PCT,
    MIN_DEPTH_MULT,
    MIN_SLICES,
    TIMEOUT_SEC,
    TWAP_DURATION_MIN,
    TWAP_ENABLED,
)
from crypto_trading_bot.utils.kraken_api import get_order_book
from crypto_trading_bot.utils.kraken_client import kraken_place_order as place_order
from crypto_trading_bot.utils.kraken_helpers import KrakenAPIError
from crypto_trading_bot.utils.price_history import ensure_min_history, get_history_prices
from crypto_trading_bot.utils.system_logger import get_system_logger

logger = get_system_logger().getChild("twap_engine")


def _sleep(seconds: float) -> None:
    if seconds <= 0 or IS_PAPER_MODE:
        return
    time.sleep(seconds)


@dataclass
class TWAPResult:
    total_filled: float
    avg_price: float
    slices_executed: int
    slippage_pct: float
    fallback_used: bool

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "total_filled": float(self.total_filled),
            "avg_price": float(self.avg_price),
            "slices_executed": int(self.slices_executed),
            "slippage_pct": float(self.slippage_pct),
            "fallback_used": bool(self.fallback_used),
        }


def _volatility_scale(pair: str) -> float:
    """Return volatility multiplier derived from recent one-minute candle std dev."""

    try:
        ensure_min_history(pair, min_len=20)
        prices = get_history_prices(pair, min_len=20)
    except Exception as exc:  # pragma: no cover - historical fallback
        logger.debug("TWAP volatility history missing for %s: %s", pair, exc)
        prices = []

    if not prices or len(prices) < 5:
        return 1.0

    arr = np.asarray(prices[-20:], dtype=np.float64)
    std = float(np.std(arr))
    mean = max(float(np.mean(arr)), 1e-9)
    vol_pct = min(std / mean, 0.02)  # cap at 2%
    return 1.0 + vol_pct * 50  # scale in range ~1 – 2


def _determine_slices(size: float, pair: str) -> int:
    if size <= 0:
        raise ValueError("TWAP size must be positive")
    vol_scale = _volatility_scale(pair)
    baseline = int(round(MIN_SLICES * vol_scale))
    return max(MIN_SLICES, min(MAX_SLICES, baseline))


def _sweetspot_ok(order_book: Dict, side: str, slice_size: float) -> bool:
    """Return True when spread and depth conditions favour maker execution."""

    if not order_book:
        return False
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if not bids or not asks:
        return False

    best_bid_price, _ = bids[0]
    best_ask_price, _ = asks[0]
    spread_pct = (best_ask_price - best_bid_price) / best_bid_price
    if spread_pct > MAX_SPREAD_PCT:
        return False

    depth_side = bids if side == "sell" else asks
    cumulative = 0.0
    for _, qty in depth_side:
        cumulative += qty
        if cumulative >= slice_size * MIN_DEPTH_MULT:
            return True
    return False


def _make_limit_order(pair: str, side: str, price: float, size: float) -> Dict[str, str]:
    return {
        "pair": pair,
        "type": side,
        "ordertype": "limit",
        "price": str(round(price, 6)),
        "volume": str(round(size, 8)),
        "oflags": "post",  # maker-only
    }


def _execute_slice(pair: str, side: str, slice_size: float, limit_price: float) -> float:
    if IS_PAPER_MODE:
        return slice_size

    order_payload = _make_limit_order(pair, side, limit_price, slice_size)
    try:
        result = place_order(**order_payload)
    except KrakenAPIError as exc:  # pragma: no cover - network dependent
        logger.warning("TWAP order failed for %s: %s", pair, exc)
        return 0.0

    filled_volume = float(result.get("filled_volume", slice_size))
    return min(slice_size, max(0.0, filled_volume))


def execute_twap(pair: str, size: float, side: str) -> TWAPResult:
    """Run TWAP execution with sweet spot logic; fallback to market after timeout."""

    if not TWAP_ENABLED:
        logger.info("TWAP disabled via configuration; skipping maker strategy")
        return TWAPResult(0.0, 0.0, 0, 0.0, False)

    side = side.lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")

    target_slices = _determine_slices(size, pair)
    slice_size = size / target_slices
    filled_total = 0.0
    cost_total = 0.0
    slices_executed = 0
    fallback_used = False
    start_time = datetime.now(timezone.utc)
    timeout_at = start_time + timedelta(seconds=TIMEOUT_SEC)

    while filled_total < size and datetime.now(timezone.utc) < timeout_at:
        order_book = get_order_book(pair)
        if not order_book:
            _sleep(5)
            continue

        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            _sleep(5)
            continue

        best_bid_price = bids[0][0]
        best_ask_price = asks[0][0]
        limit_price = best_bid_price if side == "sell" else best_ask_price

        if not _sweetspot_ok(order_book, side, slice_size):
            _sleep(max(1.0, (TWAP_DURATION_MIN * 60) / target_slices))
            continue

        filled = _execute_slice(pair, side, slice_size, limit_price)
        if filled <= 0:
            _sleep(2)
            continue

        filled_total += filled
        cost_total += filled * limit_price
        slices_executed += 1
        if filled_total >= size:
            break
        _sleep(max(1.0, (TWAP_DURATION_MIN * 60) / target_slices))

    reference_price = best_ask_price if side == "buy" else best_bid_price

    if filled_total < size:
        remaining = size - filled_total
        try:
            if IS_PAPER_MODE:
                filled_market = remaining
                avg_market_price = reference_price
            else:
                market_payload = {
                    "pair": pair,
                    "type": side,
                    "ordertype": "market",
                    "volume": str(round(remaining, 8)),
                }
                result = place_order(**market_payload)
                filled_market = float(result.get("filled_volume", remaining))
                avg_market_price = float(result.get("price", reference_price))
            filled_total += filled_market
            cost_total += filled_market * avg_market_price
            fallback_used = True
        except KrakenAPIError as exc:  # pragma: no cover - defensive
            logger.error("Market fallback failed for %s: %s", pair, exc)

    avg_price = (cost_total / filled_total) if filled_total > 0 else 0.0
    if reference_price:
        slippage_pct = (avg_price - reference_price) / reference_price
        if side == "sell":
            slippage_pct *= -1
    else:
        slippage_pct = 0.0

    return TWAPResult(
        total_filled=filled_total,
        avg_price=avg_price,
        slices_executed=slices_executed,
        slippage_pct=slippage_pct,
        fallback_used=fallback_used,
    )


def execute(pair: str, size: float, side: str) -> Dict[str, float | int | bool]:
    """Wrapper returning dictionary form result."""

    return execute_twap(pair, size, side).to_dict()
