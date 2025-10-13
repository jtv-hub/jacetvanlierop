"""
force_test_trade.py

Trigger a one-off, paper-only trade evaluation for a given pair by
injecting a synthetic RSI-friendly history and invoking the bot's
``evaluate_signals_and_trade`` flow.

Usage:
    python scripts/force_test_trade.py --pair ETH/USDC [--debug]
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone

from crypto_trading_bot.bot.trading_logic import (
    CONFIG,
    evaluate_signals_and_trade,
    ledger,
)
from crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler
from crypto_trading_bot.utils.kraken_pairs import ensure_usdc_pair
from crypto_trading_bot.utils.price_feed import get_current_price
from crypto_trading_bot.utils.price_history import (
    append_live_price,
    ensure_min_history,
    get_history_prices,
)


def _setup_logger(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("force_test_trade")
    if not logger.handlers:
        logger.addHandler(get_rotating_handler("force_test_trade.log"))
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.propagate = False
    return logger


def _inject_buy_favored_history(pair: str, current: float, period: int) -> None:
    """Append a short, steadily decreasing tail to bias RSI below lower."""
    # Ensure we have a minimum base series first
    min_needed = max(int(period) + 1, 14)
    ensure_min_history(pair, min_len=min_needed)

    # Create N descending points ending near current price to push RSI lower
    n = max(min_needed, 20)
    base = float(current)
    start = base * 1.03  # start a bit above current, then descend
    now = datetime.now(timezone.utc)
    for i in range(n):
        # Linear descent from +3% to 0%
        frac = (n - i) / n
        px = start - (start - base) * (1 - frac)
        ts = (now - timedelta(minutes=2 * (n - i))).isoformat()
        append_live_price(pair, float(px), ts=ts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Force a test trade in paper mode")
    parser.add_argument("--pair", default="ETH/USDC", help="Trading pair, e.g., ETH/USDC")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logger = _setup_logger(args.debug)

    # Enforce paper-only mode (no live keys, no live orders)
    os.environ.setdefault("ENV", "paper")
    os.environ.setdefault("DEBUG_MODE", "1" if args.debug else "0")

    pair = ensure_usdc_pair(args.pair.upper())
    price = get_current_price(pair)
    if price is None or price <= 0:
        logger.error("No current price available for %s — aborting", pair)
        print("❌ Failed: could not fetch current price")
        return 1

    # Inject a history tail that encourages a BUY signal for RSI strategy
    rsi_period = CONFIG.get("rsi", {}).get("period", 21)
    _inject_buy_favored_history(pair, float(price), int(rsi_period))

    # Snapshot trade count prior to evaluation
    before_count = len(ledger.trades or [])

    # Evaluate only the requested pair
    try:
        evaluate_signals_and_trade(check_exits_only=False, tradable_pairs=[pair])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("evaluate_signals_and_trade failed: %s", exc)
        print("❌ Failed: evaluation raised exception; see logs")
        return 2

    after_count = len(ledger.trades or [])
    if after_count > before_count:
        print(f"✅ Success: simulated trade evaluated and logged for {pair}")
        logger.info("Simulated trade logged for %s (before=%s, after=%s)", pair, before_count, after_count)
        # Optional: print last few prices used
        prices = get_history_prices(pair, min_len=int(rsi_period) + 1)
        logger.debug("Recent prices used for RSI: %s", prices[-10:])
        return 0

    print(f"⚠️ No trade was logged for {pair}. Check thresholds and volume.")
    logger.warning("No trade logged for %s (before=%s, after=%s)", pair, before_count, after_count)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
