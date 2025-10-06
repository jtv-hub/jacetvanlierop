"""Helpers for comparing paper vs live-dry trading behaviour."""

from __future__ import annotations

import logging
from typing import Dict, List

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.context.trading_context import TradingContext
from crypto_trading_bot.utils.price_feed import get_current_price
from crypto_trading_bot.utils.price_history import append_live_price, get_history_prices

from . import trading_logic

logger = logging.getLogger(__name__)


def collect_signal_snapshot(pairs: List[str]) -> List[Dict[str, object]]:
    """Return a list of strategy signals for each pair without executing trades."""

    context = TradingContext()
    context.update_context()

    rsi_period = int(CONFIG.get("rsi", {}).get("period", 21))
    min_needed = max(rsi_period + 1, 30)
    per_asset_params = CONFIG.get("strategy_params", {})
    min_volume = float(CONFIG.get("min_volume", 100.0))

    snapshots: List[Dict[str, object]] = []
    for pair in pairs:
        asset = pair.split("/")[0]
        price_now = get_current_price(pair)
        if price_now is not None and price_now > 0:
            append_live_price(pair, float(price_now))
        prices = [p for p in get_history_prices(pair, min_len=min_needed) if p is not None]
        if len(prices) < min_needed:
            logger.debug(
                "[SIM] Skipping %s — insufficient history (%d of %d)",
                pair,
                len(prices),
                min_needed,
            )
            snapshots.append(
                {
                    "pair": pair,
                    "status": "insufficient_history",
                    "signals": [],
                    "history_length": len(prices),
                    "history_required": min_needed,
                }
            )
            continue

        adx_val = context.get_adx(pair, prices)
        volume_estimate = max(min_volume, min_volume)

        strategies = trading_logic.get_strategy_pipeline(per_asset_params)

        signal_entries: List[Dict[str, object]] = []
        for strategy in strategies:
            try:
                try:
                    result = strategy.generate_signal(
                        prices,
                        volume=volume_estimate,
                        asset=asset,
                        adx=adx_val,
                    )
                except TypeError:
                    result = strategy.generate_signal(prices, volume=volume_estimate)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                signal_entries.append(
                    {
                        "strategy": strategy.__class__.__name__,
                        "error": str(exc),
                    }
                )
                continue
            signal_entries.append(
                {
                    "strategy": strategy.__class__.__name__,
                    "signal": result.get("signal") or result.get("side"),
                    "confidence": float(result.get("confidence", 0.0) or 0.0),
                }
            )

        snapshots.append(
            {
                "pair": pair,
                "signals": signal_entries,
                "history_length": len(prices),
                "history_required": min_needed,
            }
        )

    return snapshots
