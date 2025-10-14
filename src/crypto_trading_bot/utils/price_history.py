"""
price_history.py

Lightweight in-memory price history cache with seeded fallback.

Loads historical close prices from ``data/seeded_prices.json`` on demand
and exposes helpers to ensure a minimum window length for indicators
like RSI. Live prices can be appended and will not be overwritten.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

try:
    # Testing-only utility; used here as a last-resort fallback when
    # seeded history is unavailable and we need sufficient candles to proceed.
    from .mock_data_utils import generate_mock_data as _generate_mock_data  # type: ignore
except ImportError:  # pragma: no cover - optional import
    _generate_mock_data = None  # type: ignore

from crypto_trading_bot.bot.utils.alerts import send_alert
from crypto_trading_bot.config import CONFIG, is_live
from crypto_trading_bot.utils.kraken_api import get_ohlc_data

logger = logging.getLogger(__name__)

SEED_PATH = "data/seeded_prices.json"

# pair -> list of (iso_timestamp, close)
_history: Dict[str, List[Tuple[str, float]]] = {}
_seed_cache: Dict[str, List[Tuple[str, float]]] = {}


class HistoryUnavailable(RuntimeError):
    """Raised when live trading cannot proceed due to missing price history."""


_LIVE_HISTORY_ATTEMPTS = 3
_LIVE_HISTORY_BACKOFF_SECONDS = 0.75
_LIVE_HISTORY_LIMIT = 180

_fallback_metrics: Dict[str, Dict[str, int] | float | None] = {
    "mock": {},
    "live_block": {},
    "_last_event": None,
}


def _record_fallback_event(bucket: str, pair: str) -> None:
    counters = _fallback_metrics.setdefault(bucket, {})
    if isinstance(counters, dict):
        counters[pair] = counters.get(pair, 0) + 1
    _fallback_metrics["_last_event"] = time.time()


def get_fallback_metrics(reset: bool = False) -> Dict[str, Dict[str, int] | float | None]:
    """Return fallback usage counters (optionally resetting them)."""

    snapshot = {
        "mock": (dict(_fallback_metrics.get("mock", {})) if isinstance(_fallback_metrics.get("mock"), dict) else {}),
        "live_block": (
            dict(_fallback_metrics.get("live_block", {}))
            if isinstance(_fallback_metrics.get("live_block"), dict)
            else {}
        ),
        "last_event_ts": _fallback_metrics.get("_last_event"),
    }
    if reset:
        _fallback_metrics["mock"] = {}
        _fallback_metrics["live_block"] = {}
        _fallback_metrics["_last_event"] = None
    return snapshot


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_seed_file() -> None:
    if _seed_cache or not os.path.exists(SEED_PATH):
        return
    try:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pair, rows in (data or {}).items():
            out: List[Tuple[str, float]] = []
            for r in rows or []:
                if not isinstance(r, (list, tuple)) or len(r) < 2:
                    continue
                ts = str(r[0])
                try:
                    price = float(r[1])
                except (ValueError, TypeError):
                    continue
                out.append((ts, price))
            if out:
                _seed_cache[pair.upper()] = out
    except (OSError, json.JSONDecodeError):
        # Fail-open: simply keep empty cache
        _seed_cache.clear()


def _attempt_live_history(pair: str, min_len: int) -> List[Tuple[str, float]]:
    """Retry Kraken OHLC before resorting to mock data so live mode only runs on real candles."""
    key = pair.upper()
    target = max(min_len, _LIVE_HISTORY_LIMIT)
    for attempt in range(_LIVE_HISTORY_ATTEMPTS):
        try:
            rows = get_ohlc_data(pair, interval=1, limit=target)
        except Exception as exc:  # pragma: no cover - network dependent  # pylint: disable=broad-exception-caught
            logger.warning(
                "Live history fetch attempt %d/%d failed for %s: %s",
                attempt + 1,
                _LIVE_HISTORY_ATTEMPTS,
                pair,
                exc,
            )
            if attempt < _LIVE_HISTORY_ATTEMPTS - 1:
                time.sleep(_LIVE_HISTORY_BACKOFF_SECONDS * (attempt + 1))
            continue

        if not rows:
            logger.debug("Kraken OHLC returned no rows for %s", pair)
            continue

        normalized: List[Tuple[str, float]] = []
        for row in rows:
            ts = row.get("time")
            close_px = row.get("close")
            if ts is None or close_px is None:
                continue
            try:
                iso_ts = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                normalized.append((iso_ts, float(close_px)))
            except (TypeError, ValueError, OSError):
                continue

        if normalized:
            # Keep most recent candles only to bound memory usage
            normalized.sort(key=lambda item: item[0])
            _history[key] = normalized[-target:]
            logger.info(
                "Loaded %d live candles for %s from Kraken OHLC",
                len(_history[key]),
                pair,
            )
            return _history[key]

        if attempt < _LIVE_HISTORY_ATTEMPTS - 1:
            time.sleep(_LIVE_HISTORY_BACKOFF_SECONDS * (attempt + 1))

    return _history.get(key, [])


def ensure_min_history(pair: str, min_len: int = 14) -> None:
    """Ensure ``_history[pair]`` has at least ``min_len`` entries.

    If not, load from the seed file and inject the seeded rows that are not
    already present by timestamp. Logs once per injection.
    """
    if not isinstance(pair, str) or "/" not in pair:
        return
    key = pair.upper()
    cur = _history.get(key, [])
    if len(cur) >= min_len:
        _history[key] = cur
        return

    _load_seed_file()
    seeded = list(_seed_cache.get(key, []))
    if not seeded:
        _history[key] = cur
        return

    # Avoid duplicates by timestamp
    seen = {ts for ts, _ in cur}
    injected = [(ts, px) for ts, px in seeded if ts not in seen]
    if injected:
        # Keep chronological order; seed data assumed oldest->newest
        cur = (cur + injected)[-max(min_len, len(cur) + len(injected)) :]
        _history[key] = cur
        print(f"[SEED FALLBACK] Injected seeded history for {key} ({len(injected)} candles)")
    else:
        _history[key] = cur


def append_live_price(pair: str, price: float, ts: str | None = None) -> None:
    """Append a live price with timestamp; avoid simple duplicates.

    If the last timestamp equals ``ts`` or the last price equals ``price``,
    do not append a duplicate entry.
    """
    if not pair or price is None:
        return
    key = pair.upper()
    ts_iso = ts or _now_iso()
    cur = _history.setdefault(key, [])
    if cur:
        last_ts, last_px = cur[-1]
        if last_ts == ts_iso or (last_px == price):
            # Update last timestamp if same price but different time
            if last_ts != ts_iso and last_px == price:
                cur[-1] = (ts_iso, price)
            return
    cur.append((ts_iso, float(price)))


def get_history_prices(pair: str, min_len: int = 14) -> List[float]:
    """Return a list of closes for ``pair`` ensuring ``min_len`` via seeds.

    Adds explicit debug logging and a safety mock fallback (30 candles)
    if seeded data is missing and fewer than 15 valid candles are available.
    """
    req = max(int(min_len), 1)
    ensure_min_history(pair, min_len=req)
    key = pair.upper()
    rows = _history.get(key, [])
    prices = [px for _, px in rows]
    valid_prices = [px for px in prices if px is not None]
    print(f"[DEBUG] Requested {req} candles for {pair}")
    print(f"[DEBUG] Fetched {len(valid_prices)} valid candles for {pair}")

    live_real_mode = is_live and not bool(CONFIG.get("live_mode", {}).get("dry_run"))

    if len(valid_prices) < req:
        refreshed = _attempt_live_history(pair, req)
        if refreshed:
            prices = [px for _, px in refreshed]
            valid_prices = [px for px in prices if px is not None]
            print(f"[LIVE HISTORY] Pulled {len(valid_prices)} candles for {pair} after live retry")

    if len(valid_prices) < req and live_real_mode:
        _record_fallback_event("live_block", pair.upper())
        message = f"Live trading blocked for {pair}: insufficient candles " f"(need {req}, have {len(valid_prices)})."
        context = {"pair": pair, "required": req, "available": len(valid_prices)}
        send_alert(message, context=context, level="CRITICAL")
        logger.critical(message)
        raise HistoryUnavailable(message)

    if len(valid_prices) < 15:
        print(f"[ERROR] Insufficient candles fetched for {pair}: only {len(valid_prices)}")
        if _generate_mock_data is not None:
            print(f"[MOCK DATA] Using mock prices for {pair} due to fetch failure")
            _record_fallback_event("mock", pair.upper())
            mock_pair = pair.replace("/", "-")
            try:
                snap = _generate_mock_data(mock_pair)
                base_price = float(snap.get("price", 100.0))
            except Exception:  # pylint: disable=broad-exception-caught
                base_price = 100.0
            steps = max(req, 30)
            px = base_price
            series: List[Tuple[str, float]] = []
            for _ in range(steps):
                drift = random.uniform(-0.005, 0.005)
                px = max(0.01, px * (1.0 + drift))
                series.append((_now_iso(), round(px, 6)))
            _history[key] = series
            logger.warning(
                "Mock price series injected for %s (steps=%d base=%.4f)",
                pair,
                steps,
                base_price,
            )
            return [p for _, p in series]

    return valid_prices


__all__ = [
    "ensure_min_history",
    "append_live_price",
    "get_history_prices",
    "HistoryUnavailable",
    "get_fallback_metrics",
]
