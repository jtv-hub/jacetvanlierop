"""
Lightweight Kraken REST API utility.

- Public Ticker endpoint only (no auth required)
- Converts app pairs like "BTC/USD" to Kraken's altname (e.g., "XBTUSD")
- Safe retries with backoff and structured logging
- Designed to be imported by other modules, e.g., price feeds

Example:
    from crypto_trading_bot.utils.kraken_api import get_ticker_price
    px = get_ticker_price("BTC/USD")
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional

# Optional HTTP clients (prefer requests, fallback to httpx)
try:  # pragma: no cover - optional dependency
    import requests as _requests  # type: ignore

    _HAVE_REQUESTS = True
except ImportError:  # pragma: no cover - optional dependency
    _requests = None  # type: ignore[assignment]
    _HAVE_REQUESTS = False

try:  # pragma: no cover - optional dependency
    import httpx as _httpx  # type: ignore

    _HAVE_HTTPX = True
except ImportError:  # pragma: no cover - optional dependency
    _httpx = None  # type: ignore[assignment]
    _HAVE_HTTPX = False

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.utils.kraken_pairs import PAIR_MAP, normalize_pair

logger = logging.getLogger(__name__)

# App format -> Kraken altname format. Include slash-based keys to be explicit.
# Kraken uses "XBT" instead of "BTC". Others typically map 1:1.
# Public endpoints (no auth required)
_TICKER_URL = "https://api.kraken.com/0/public/Ticker"
_OHLC_URL = "https://api.kraken.com/0/public/OHLC"


def _normalize_pair(pair: str) -> str:
    """Return Kraken altname for a human pair like "BTC/USDC".

    - Uses PAIR_MAP if present; otherwise swaps BTC->XBT and removes slash.
    - Uppercases input defensively.
    """
    if not isinstance(pair, str) or "/" not in pair:
        raise ValueError(f"Invalid pair format: {pair!r}; expected like 'BTC/USD'")
    return normalize_pair(pair)


def _build_headers() -> Dict[str, str]:
    """Return default headers for Kraken requests.

    API key/secret are not used for public endpoints, but we include
    a UA for friendlier logs on the exchange side.
    """
    headers = {
        "User-Agent": "crypto-trading-bot/kraken-api (+https://example.com)",
        "Accept": "application/json",
    }
    # Reserved for future private endpoints:
    _ = CONFIG.get("kraken_api_key")
    _ = CONFIG.get("kraken_api_secret")
    return headers


def _http_get(url: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Perform an HTTP GET with either requests or httpx and return JSON.

    Raises RuntimeError on transport-level failures so callers can retry.
    """
    headers = _build_headers()

    if _HAVE_REQUESTS and _requests is not None:
        try:
            r = _requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (
            _requests.exceptions.RequestException,  # type: ignore[attr-defined]
            ValueError,
            TypeError,
        ) as exc:
            raise RuntimeError(f"HTTP GET failed: {exc}") from exc
    if _HAVE_HTTPX and _httpx is not None:
        try:
            with _httpx.Client(timeout=timeout, headers=headers) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                return r.json()
        except (_httpx.HTTPError, ValueError, TypeError) as exc:  # type: ignore[attr-defined]
            raise RuntimeError(f"HTTP GET failed: {exc}") from exc
    raise RuntimeError("No HTTP client available. Install 'requests' or 'httpx'.")


def _extract_last_price(payload: Dict[str, Any]) -> float:
    """Extract last price from Kraken Ticker response.

    Kraken returns a shape like:
        {"error": [], "result": {"XBTUSD": {"c": ["117333.1", "1.0"], ...}}}

    We take the first entry under result and prefer 'c'[0] (last trade close),
    falling back to ask/bid if needed.
    """
    if not isinstance(payload, dict):
        raise ValueError("Ticker payload is not a dict")

    if payload.get("error"):
        raise RuntimeError(f"Kraken error(s): {payload['error']}")

    result = payload.get("result")
    if not isinstance(result, dict) or not result:
        raise ValueError("Ticker payload missing 'result'")

    first_entry = next(iter(result.values()))
    if not isinstance(first_entry, dict):
        raise ValueError("Unexpected ticker entry format")

    def _first_float(seq: Any) -> Optional[float]:
        try:
            return float(seq[0])
        except (TypeError, ValueError, IndexError):
            return None

    last = _first_float(first_entry.get("c"))
    if last is not None:
        return last

    # Fallbacks in rare cases
    ask = _first_float(first_entry.get("a"))
    bid = _first_float(first_entry.get("b"))
    if ask is not None:
        return ask
    if bid is not None:
        return bid

    raise ValueError("Ticker payload missing price fields (c/a/b)")


def _extract_ohlc_rows(payload: Dict[str, Any], pair: str) -> list[dict[str, float]]:
    """Normalize OHLC rows from Kraken payload."""

    if not isinstance(payload, dict):
        raise ValueError("OHLC payload is not a dict")

    if payload.get("error"):
        raise RuntimeError(f"Kraken error(s): {payload['error']}")

    result = payload.get("result")
    if not isinstance(result, dict) or not result:
        raise ValueError("OHLC payload missing 'result'")

    raw_rows = result.get(pair)
    if raw_rows is None:
        for key, value in result.items():
            if key != "last":
                raw_rows = value
                break
    if raw_rows is None:
        raise ValueError(f"OHLC payload missing rows for {pair}")
    if not isinstance(raw_rows, list):
        raise ValueError("OHLC rows are not a list")

    normalized: list[dict[str, float]] = []
    for row in raw_rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        try:
            ts = int(row[0])
        except (TypeError, ValueError):
            continue
        try:
            open_px = float(row[1])
            high_px = float(row[2])
            low_px = float(row[3])
            close_px = float(row[4])
            volume = float(row[6]) if len(row) > 6 else 0.0
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                "time": ts,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "volume": volume,
            }
        )
    return normalized


def get_ohlc_data(
    pair: str,
    *,
    interval: int = 1,
    since: int | None = None,
    limit: int = 120,
    timeout: float = 10.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> list[dict[str, float]]:
    """Fetch OHLC candles for ``pair`` from Kraken public API."""

    kpair = _normalize_pair(pair)
    params: Dict[str, Any] = {"pair": kpair, "interval": max(1, int(interval))}
    if since is not None:
        params["since"] = int(since)

    attempt = 0
    while True:
        try:
            logger.debug(
                "Kraken GET %s params=%s attempt=%d",
                _OHLC_URL,
                params,
                attempt + 1,
            )
            payload = _http_get(_OHLC_URL, params, timeout)
            rows = _extract_ohlc_rows(payload, kpair)
            if not rows:
                raise ValueError("OHLC response empty")
            if limit > 0:
                rows = rows[-min(len(rows), limit) :]
            logger.info(
                "Kraken OHLC fetched %d candles for %s (interval=%s)",
                len(rows),
                pair,
                params["interval"],
            )
            return rows
        except (RuntimeError, ValueError) as exc:
            if attempt >= max(0, retries - 1):
                logger.error("Kraken OHLC failed for %s (%s): %s", pair, kpair, exc)
                raise
            sleep_for = backoff * (2**attempt) + random.uniform(0.0, 0.25)
            logger.warning(
                "Kraken OHLC error (attempt %d/%d) for %s (%s): %s; retrying in %.2fs",
                attempt + 1,
                retries,
                pair,
                kpair,
                exc,
                sleep_for,
            )
            time.sleep(max(0.05, sleep_for))
            attempt += 1


def get_ticker_price(
    pair: str,
    *,
    timeout: float = 10.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> float:
    """Return the latest market price for a given pair like "BTC/USD".

    - Uses Kraken public /Ticker endpoint
    - Retries transient network or response errors with exponential backoff
    - Logs at DEBUG level for traceability; INFO on success
    - Raises a RuntimeError or ValueError if no price can be obtained
    """
    kpair = _normalize_pair(pair)
    params = {"pair": kpair}

    attempt = 0
    while True:
        try:
            logger.debug("Kraken GET %s params=%s attempt=%d", _TICKER_URL, params, attempt + 1)
            payload = _http_get(_TICKER_URL, params, timeout)
            price = _extract_last_price(payload)
            logger.info("Kraken price %s (%s): %s", pair, kpair, price)
            return price
        except (RuntimeError, ValueError) as exc:
            if attempt >= max(0, retries - 1):
                logger.error("Kraken ticker failed for %s (%s): %s", pair, kpair, exc)
                raise
            sleep_for = backoff * (2**attempt) + random.uniform(0.0, 0.25)
            logger.warning(
                "Kraken ticker error (attempt %d/%d) for %s (%s): %s; retrying in %.2fs",
                attempt + 1,
                retries,
                pair,
                kpair,
                exc,
                sleep_for,
            )
            time.sleep(max(0.05, sleep_for))
            attempt += 1


try:  # Lazy import to avoid circular dependency during module init
    from crypto_trading_bot.utils.kraken_client import kraken_client
except ImportError:  # pragma: no cover - fallback if private client unavailable
    kraken_client = None  # type: ignore[assignment]

__all__ = ["get_ticker_price", "get_ohlc_data", "PAIR_MAP", "kraken_client"]
