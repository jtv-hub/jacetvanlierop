"""Evaluate open positions and trigger exits when conditions are met."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, cast

from ..bot.trading_logic import PositionManager
from ..ledger.trade_ledger import TradeLedger
from ..utils.file_locks import _locked_file
from ..utils.price_feed import get_current_price

if __name__ == "__main__" and __package__ is None:
    print(
        ("This module must be executed as " "'python -m crypto_trading_bot.scripts.check_exit_conditions'."),
        file=sys.stderr,
    )
    sys.exit(1)

LOGGER = logging.getLogger(__name__)
if not LOGGER.hasHandlers():  # pragma: no cover - logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

STALE_PRICE_THRESHOLD = timedelta(minutes=5)
TRADES_LOG_PATH = "logs/trades.log"
EXIT_CHECK_LOG_PATH = "logs/exit_check.log"


def _normalise_pair(pair: object) -> str | None:
    """Return an upper-cased trading pair string or None."""

    if isinstance(pair, str):
        stripped = pair.strip()
        if stripped:
            return stripped.upper()
    return None


def _iter_pairs(positions: Iterable[dict]) -> set[str]:
    """Collect the distinct trading pairs in the given positions."""

    return {pair for pos in positions if (pair := _normalise_pair(pos.get("pair"))) is not None}


def _normalise_timestamp(raw_ts: object) -> datetime | None:
    """Convert supported timestamp formats to timezone-aware datetimes."""

    if isinstance(raw_ts, datetime):
        return raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
    if isinstance(raw_ts, (int, float)) and raw_ts > 0:
        return datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
    return None


def _fetch_price_with_timestamp(pair: str) -> tuple[float | None, datetime | None]:
    """Fetch the latest price and timestamp for the requested pair."""

    price_fetcher = cast(Any, get_current_price)
    try:
        # pylint: disable=unexpected-keyword-arg
        price_result = price_fetcher(
            pair,
            return_timestamp=True,
        )
        # pylint: enable=unexpected-keyword-arg
    except TypeError:
        # Older API without return_timestamp support.
        price_result = price_fetcher(pair)
        timestamp = datetime.now(timezone.utc) if price_result is not None else None
    else:
        if isinstance(price_result, tuple) and len(price_result) == 2:
            price_result, raw_timestamp = price_result
            timestamp = _normalise_timestamp(raw_timestamp)
        else:
            timestamp = datetime.now(timezone.utc) if price_result is not None else None

    try:
        price = float(price_result) if price_result is not None else None
    except (TypeError, ValueError):
        price = None

    return price if (price is not None and price > 0) else None, timestamp


def _entry_price_fallback(pm: PositionManager, pair: str) -> float | None:
    """Return the first valid entry price for ``pair`` from open positions."""

    for pos in pm.positions.values():
        if _normalise_pair(pos.get("pair")) != pair:
            continue
        try:
            entry_price = float(pos.get("entry_price"))
        except (TypeError, ValueError):
            entry_price = None
        if entry_price and entry_price > 0:
            return entry_price
    return None


def build_current_prices(
    pm: PositionManager,
) -> tuple[dict[str, float], dict[str, datetime | None]]:
    """Fetch latest prices for all pairs in the open positions.

    Falls back to position entry_price when live data is unavailable or stale.
    """

    prices: dict[str, float] = {}
    timestamps: dict[str, datetime | None] = {}
    pairs = sorted(_iter_pairs(pm.positions.values()))
    now = datetime.now(timezone.utc)

    for pair in pairs:
        live_price, price_timestamp = _fetch_price_with_timestamp(pair)
        timestamp = _normalise_timestamp(price_timestamp)
        is_stale = timestamp is None or (now - timestamp) > STALE_PRICE_THRESHOLD

        if live_price is not None and not is_stale:
            prices[pair] = float(live_price)
            timestamps[pair] = timestamp
            LOGGER.debug("Fetched live price for %s: %.6f", pair, live_price)
            continue

        fallback_price = _entry_price_fallback(pm, pair)
        if fallback_price is not None:
            prices[pair] = fallback_price
            timestamps[pair] = None
            if live_price is None:
                LOGGER.warning("Using entry price fallback for %s: live price unavailable", pair)
            else:
                LOGGER.warning("Using entry price fallback for %s: price stale", pair)
        else:
            LOGGER.warning("No live price or fallback for %s; skipping in checks", pair)

    return prices, timestamps


def _sync_trades_log() -> None:
    """Ensure the trades log is flushed to disk with cooperative locking."""

    os.makedirs(os.path.dirname(TRADES_LOG_PATH), exist_ok=True)
    with _locked_file(TRADES_LOG_PATH, "a") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _safe_entry_price(position: dict | None) -> float | None:
    """Return a validated positive entry price from the position, if possible."""

    if not position:
        return None
    try:
        entry_price = float(position.get("entry_price"))
    except (TypeError, ValueError):
        return None
    return entry_price if entry_price > 0 else None


def _append_exit_record(record: dict) -> None:
    """Append a JSON record to the exit check log using an exclusive lock."""

    os.makedirs(os.path.dirname(EXIT_CHECK_LOG_PATH), exist_ok=True)
    with _locked_file(EXIT_CHECK_LOG_PATH, "a") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:  # pylint: disable=too-many-locals
    """Execute a single exit evaluation cycle."""

    position_manager = PositionManager()
    position_manager.load_positions_from_file()
    if not position_manager.positions:
        LOGGER.info("No open positions found; exit checks complete.")
        return 0

    ledger = TradeLedger(position_manager)
    ledger.reload_trades()

    prices, price_timestamps = build_current_prices(position_manager)
    if not prices:
        LOGGER.info("No price data available; skipping exit checks.")
        return 0

    # Preserve pre-exit state for ROI calculation (check_exits mutates positions).
    position_snapshot = {trade_id: dict(position) for trade_id, position in position_manager.positions.items()}

    check_exits_callable = cast(Any, position_manager.check_exits)
    try:
        # pylint: disable=unexpected-keyword-arg
        exits = check_exits_callable(
            prices,
            price_timestamps=price_timestamps,
        )
        # pylint: enable=unexpected-keyword-arg
    except TypeError:
        exits = check_exits_callable(prices)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("Exit evaluation failed: %s", exc)
        return 1

    for trade_id, exit_price, reason in exits:
        LOGGER.info("Closing trade %s at %.6f: %s", trade_id, exit_price, reason)
        try:
            ledger.update_trade(trade_id=trade_id, exit_price=exit_price, reason=reason)
            _sync_trades_log()

            position = position_snapshot.get(trade_id, {})
            entry_price = _safe_entry_price(position)
            try:
                exit_price_f = float(exit_price)
            except (TypeError, ValueError):
                exit_price_f = None

            roi = None
            if entry_price and exit_price_f and entry_price != 0:
                roi = (exit_price_f - entry_price) / entry_price

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_id": trade_id,
                "exit_price": round(exit_price_f, 8) if exit_price_f is not None else None,
                "reason": reason,
                "roi": round(roi, 8) if isinstance(roi, float) else None,
            }
            _append_exit_record(record)
        except (OSError, IOError, ValueError) as exc:
            LOGGER.exception("Failed to update trade %s: %s", trade_id, exc)

    if not exits:
        LOGGER.info("No exit conditions triggered.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - final safeguard
        # pylint: disable=broad-exception-caught
        LOGGER.exception("Fatal error during exit check: %s", exc)
        sys.exit(1)
