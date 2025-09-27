"""Kraken private REST client helpers.

This module implements minimal authenticated endpoints required for live
trading: balance lookups, order submission, optional cancellation, and
open-order inspection. It relies on API credentials provided via
environment variables loaded in ``crypto_trading_bot.config``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlencode

try:  # pragma: no cover - optional dependency
    import requests

    _HAVE_REQUESTS = True
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]
    _HAVE_REQUESTS = False

try:  # pragma: no cover - optional dependency
    import httpx  # type: ignore[import-not-found]

    _HAVE_HTTPX = True
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore[assignment]
    _HAVE_HTTPX = False

from crypto_trading_bot.config import CONFIG, _sanitize_base64_secret

try:  # Reuse pair normalization rules from the public client when available.
    from crypto_trading_bot.utils import kraken_api as _public_api
except ImportError:  # pragma: no cover - fallback if module renamed/removed
    _public_api = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_API_BASE = "https://api.kraken.com"
_PRIVATE_PREFIX = "/0/private/"
_PUBLIC_PREFIX = "/0/public/"
_USER_AGENT = "crypto-trading-bot/1.0 (Kraken private client)"

# Kraken sometimes prefixes assets with X/Z on the private balance endpoint.
_ASSET_HINTS: Dict[str, Iterable[str]] = {
    "USDC": ("USDC", "ZUSD", "USDT"),  # include close substitutes for resiliency
    "USD": ("ZUSD", "USD"),
    "BTC": ("XXBT", "XBT", "BTC"),
    "ETH": ("XETH", "ETH"),
    "SOL": ("SOL",),
    "LINK": ("LINK",),
    "XRP": ("XXRP", "XRP"),
}


_PAIR_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_PAIR_CACHE_TTL = 300.0


def _invalidate_pair_cache() -> None:
    _PAIR_CACHE.clear()


class KrakenAPIError(RuntimeError):
    """Raised when Kraken returns an error payload or transport fails."""


class KrakenAuthError(KrakenAPIError):
    """Raised when credentials are missing or invalid."""


def _get_credentials() -> tuple[str, str]:
    key = (CONFIG.get("kraken_api_key") or "").strip()
    secret = (CONFIG.get("kraken_api_secret") or "").strip()
    if not key or not secret:
        raise KrakenAuthError(
            "Kraken API credentials missing. Ensure KRAKEN_API_KEY and "
            "KRAKEN_API_SECRET are set in the environment or .env file.",
        )
    key_origin = CONFIG.get("_kraken_key_origin", "unknown")
    secret_origin = CONFIG.get("_kraken_secret_origin", "unknown")
    logger.debug(
        "Kraken credentials sanitised | key_prefix=%s key_length=%d key_origin=%s " "secret_length=%d secret_origin=%s",
        (key[:6] + "***") if len(key) >= 6 else "***",
        len(key),
        key_origin,
        len(secret),
        secret_origin,
    )
    logger.debug(
        "Kraken credentials fetched | key_prefix=%s key_origin=%s secret_origin=%s",
        (key[:6] + "***") if len(key) >= 6 else "***",
        key_origin,
        secret_origin,
    )
    return key, secret


def _decode_secret(secret: str) -> bytes:
    """Decode the Kraken secret, adding padding and validating base64."""

    try:
        normalized = _sanitize_base64_secret(secret or "", strict=True)
    except ValueError as exc:
        logger.error("Kraken secret sanitization failed: %s", exc)
        raise KrakenAuthError("Invalid Kraken API secret / malformed base64") from exc

    if not normalized:
        raise KrakenAuthError("Kraken API secret missing.")

    try:
        decoded = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.error("Kraken secret decode failed: %s", exc)
        raise KrakenAuthError("Invalid Kraken API secret / malformed base64") from exc

    logger.debug("Kraken secret decoded length=%d", len(decoded))
    return decoded


def _normalize_pair(pair: str) -> str:
    """Return Kraken's altname for a pair like ``BTC/USD``."""

    if _public_api and hasattr(_public_api, "PAIR_MAP"):
        pair_map = getattr(_public_api, "PAIR_MAP")
        if isinstance(pair_map, dict):
            mapped = pair_map.get(pair.upper())
            if mapped:
                return mapped
    # Fallback behaviour mirrors the public helper: uppercase, swap BTC->XBT, drop slash.
    up = pair.upper()
    if "/" not in up:
        raise ValueError(f"Invalid trading pair format: {pair!r}")
    base, quote = up.split("/", 1)
    if base == "BTC":
        base = "XBT"
    return f"{base}{quote}"


def _asset_candidates(asset: str) -> tuple[str, ...]:
    """Return likely Kraken asset codes for a human-readable asset symbol."""

    if not asset:
        return ("",)
    symbol = asset.upper()
    hints = _ASSET_HINTS.get(symbol)
    if hints:
        return tuple(dict.fromkeys(hints))  # drop duplicates, keep order
    candidates = [symbol]
    if not symbol.startswith(("X", "Z")):
        candidates.append(f"X{symbol}")
        candidates.append(f"Z{symbol}")
    return tuple(dict.fromkeys(candidates))


def _build_headers(api_key: str, api_sign: str) -> Dict[str, str]:
    return {
        "API-Key": api_key,
        "API-Sign": api_sign,
        "User-Agent": _USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "Accept": "application/json",
    }


def _sign_request(url_path: str, nonce: str, postdata: str, secret: str) -> str:
    """Return Kraken HMAC signature for a private request."""

    secret_bytes = _decode_secret(secret)
    sha = hashlib.sha256((nonce + postdata).encode()).digest()
    message = url_path.encode() + sha
    mac = hmac.new(secret_bytes, message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def _http_post(url: str, data: str, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    """Perform an HTTP POST using requests or httpx and return JSON."""

    if _HAVE_REQUESTS and requests is not None:
        try:
            response = requests.post(url, data=data, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (  # type: ignore[attr-defined]
            requests.exceptions.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            raise KrakenAPIError(f"HTTP POST failed: {exc}") from exc

    if _HAVE_HTTPX and httpx is not None:
        try:
            with httpx.Client(timeout=timeout, headers=headers) as client:
                response = client.post(url, content=data)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:  # type: ignore[attr-defined]
            raise KrakenAPIError(f"HTTP POST failed: {exc}") from exc

    raise KrakenAPIError("No HTTP client available. Install 'requests' or 'httpx'.")


def _http_public(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Perform an HTTP GET against Kraken's public REST API."""

    url = f"{_API_BASE}{_PUBLIC_PREFIX}{endpoint}"
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }

    if _HAVE_REQUESTS and requests is not None:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (  # type: ignore[attr-defined]
            requests.exceptions.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            raise KrakenAPIError(f"HTTP GET failed: {exc}") from exc

    if _HAVE_HTTPX and httpx is not None:
        try:
            with httpx.Client(timeout=timeout, headers=headers) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:  # type: ignore[attr-defined]
            raise KrakenAPIError(f"HTTP GET failed: {exc}") from exc

    raise KrakenAPIError("No HTTP client available for public requests. Install 'requests' or 'httpx'.")


def _standard_response(
    base: Dict[str, Any],
    *,
    endpoint: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize response shape for downstream consumers."""

    data = {
        "ok": bool(base.get("ok")),
        "error": base.get("error"),
        "code": base.get("code"),
        "endpoint": endpoint,
        "result": base.get("result"),
        "raw": base.get("raw"),
    }
    if extra:
        data.update(extra)
    return data


def _private_request(
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 15.0,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """Make an authenticated request to Kraken's private REST API."""

    try:
        api_key, api_secret = _get_credentials()
    except KrakenAuthError as exc:
        logger.error("Kraken credential error: %s", exc)
        return {
            "ok": False,
            "error": str(exc),
            "code": "auth",
            "endpoint": endpoint,
            "result": None,
            "raw": None,
        }

    try:
        secret_bytes = _decode_secret(api_secret)
    except KrakenAuthError as exc:
        logger.error("Kraken secret validation error: %s", exc)
        return {
            "ok": False,
            "error": str(exc),
            "code": "auth",
            "endpoint": endpoint,
            "result": None,
            "raw": None,
        }

    url_path = f"{_PRIVATE_PREFIX}{endpoint}"
    url = f"{_API_BASE}{url_path}"

    base_payload = dict(payload or {})
    attempt = 0
    last_error: str | None = None

    while attempt < max_attempts:
        attempt += 1
        payload_with_nonce = dict(base_payload)
        nonce = str(int(time.time() * 1000))
        payload_with_nonce["nonce"] = nonce
        postdata = urlencode(payload_with_nonce, doseq=True)

        try:
            api_sign = _sign_request(url_path, nonce, postdata, api_secret)
        except KrakenAuthError as exc:
            logger.error("Kraken signing error: %s", exc)
            return {
                "ok": False,
                "error": str(exc),
                "code": "auth",
                "endpoint": endpoint,
                "result": None,
                "raw": None,
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected signing error for %s: %s", endpoint, exc)
            return {
                "ok": False,
                "error": f"Signing error: {exc}",
                "code": "sign",
                "endpoint": endpoint,
                "result": None,
                "raw": None,
            }

        headers = _build_headers(api_key, api_sign)

        try:
            logger.debug(
                "Kraken POST %s data=%s attempt=%d secret_len=%d",
                url_path,
                _redact_nonce(postdata),
                attempt,
                len(secret_bytes),
            )
            http_response = _http_post(url, postdata, headers, timeout)
        except KrakenAPIError as exc:
            last_error = str(exc)
            logger.warning(
                "Kraken transport error on %s attempt %d/%d: %s",
                endpoint,
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(0.5 * attempt)
            continue
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected HTTP error for %s: %s", endpoint, exc)
            return {
                "ok": False,
                "error": f"HTTP error: {exc}",
                "code": "http",
                "endpoint": endpoint,
                "result": None,
                "raw": None,
            }

        if not isinstance(http_response, dict):
            logger.error("Unexpected Kraken response type: %s", type(http_response))
            return {
                "ok": False,
                "error": f"Unexpected response type: {type(http_response)}",
                "code": "response",
                "endpoint": endpoint,
                "result": None,
                "raw": http_response,
            }

        errors = http_response.get("error")
        if errors:
            error_list = errors if isinstance(errors, list) else [str(errors)]
            combined_error = "; ".join(error_list)
            code = "api"
            normalized_errors = [err.lower() for err in error_list]
            if any("cost minimum not met" in err for err in normalized_errors):
                code = "cost_minimum_not_met"
            elif any("order minimum not met" in err or "invalid arguments:volume" in err for err in normalized_errors):
                code = "volume_minimum_not_met"
            elif any("rate limit" in err for err in normalized_errors):
                code = "rate_limit"
                _invalidate_pair_cache()
            logger.error(
                "Kraken API error payload on %s: %s (code=%s)",
                endpoint,
                error_list,
                code,
            )
            return {
                "ok": False,
                "error": combined_error,
                "code": code,
                "endpoint": endpoint,
                "result": None,
                "raw": http_response,
                "errors": error_list,
            }

        result = http_response.get("result")
        if result is None:
            logger.error("Kraken response missing result field on %s", endpoint)
            return {
                "ok": False,
                "error": "Missing result in Kraken response",
                "code": "response",
                "endpoint": endpoint,
                "result": None,
                "raw": http_response,
            }

        return {
            "ok": True,
            "error": None,
            "code": "ok",
            "endpoint": endpoint,
            "result": result,
            "raw": http_response,
        }

    return {
        "ok": False,
        "error": last_error or "Max attempts exceeded for Kraken request",
        "code": "http",
        "endpoint": endpoint,
        "result": None,
        "raw": None,
    }


def _redact_nonce(postdata: str) -> str:
    """Hide the nonce value in debug logs."""

    if "nonce=" not in postdata:
        return postdata
    parts = []
    for kv in postdata.split("&"):
        if kv.startswith("nonce="):
            parts.append("nonce=***")
        else:
            parts.append(kv)
    return "&".join(parts)


def kraken_get_balance(asset: str = "USDC") -> Dict[str, Any]:
    """Return the available balance for ``asset`` (defaults to USDC)."""

    base = _private_request("Balance")
    response = _standard_response(base, endpoint="Balance", extra={"asset": asset.upper()})
    if not response.get("ok"):
        response.setdefault("balance", None)
        return response

    candidates = _asset_candidates(asset)
    result = response.get("result") or {}
    for key in candidates:
        if key in result:
            try:
                balance = float(result[key])
            except (TypeError, ValueError):
                continue
            logger.info("Kraken balance %s (%s): %.8f", asset.upper(), key, balance)
            response["balance"] = balance
            response["key"] = key
            return response

    message = f"Kraken balance lookup missing asset {asset}"
    logger.warning("%s (candidates=%s)", message, candidates)
    response["ok"] = False
    response["error"] = message
    response["code"] = response.get("code") or "asset"
    response["balance"] = None
    return response


def query_api_key_permissions() -> Optional[Dict[str, Any]]:
    """Return the rights/permissions associated with the current API key.

    Uses Kraken's private ``QueryKey`` endpoint. If the call fails or the
    response cannot be parsed, returns ``None`` instead of raising so callers
    can decide how strictly to enforce the check.
    """

    response = _private_request("QueryKey", {})
    normalized = _standard_response(response, endpoint="QueryKey")
    if not normalized.get("ok"):
        logger.warning(
            "Unable to query Kraken API key permissions: %s",
            normalized.get("error") or normalized.get("code"),
        )
        return None

    result = normalized.get("result")
    rights: Dict[str, Any] = {}
    if isinstance(result, dict):
        if "rights" in result:
            rights = result.get("rights") or {}
        else:
            try:
                first = next(iter(result.values()))
            except StopIteration:
                first = None
            if isinstance(first, dict):
                rights = first.get("rights", {}) or {}

    if not rights:
        logger.warning("Kraken API key permissions response missing rights payload: %s", result)

    return {"rights": rights, "raw": result}


def kraken_get_asset_pair_meta(pair: str) -> Dict[str, Any]:
    """Return Kraken asset pair metadata including order and cost minimums."""

    normalized = _normalize_pair(pair)
    now = time.monotonic()
    cached = _PAIR_CACHE.get(normalized)
    if cached and now - cached[0] < _PAIR_CACHE_TTL:
        return dict(cached[1])

    payload = _http_public("AssetPairs", {"pair": normalized})
    errors = payload.get("error")
    if errors:
        raise KrakenAPIError(f"AssetPairs error for {pair}: {errors}")

    result = payload.get("result")
    if not isinstance(result, dict):
        raise KrakenAPIError(f"Invalid AssetPairs payload for {pair}")

    meta = result.get(normalized)
    if not isinstance(meta, dict):
        raise KrakenAPIError(f"Asset metadata missing for pair {pair}")

    try:
        ordermin = float(meta.get("ordermin", 0.0) or 0.0)
    except (TypeError, ValueError):
        ordermin = 0.0
    try:
        costmin = float(meta.get("costmin", 0.0) or 0.0)
    except (TypeError, ValueError):
        costmin = 0.0
    try:
        price_decimals = int(meta.get("pair_decimals", 5) or 5)
    except (TypeError, ValueError):
        price_decimals = 5
    try:
        lot_decimals = int(meta.get("lot_decimals", 8) or 8)
    except (TypeError, ValueError):
        lot_decimals = 8

    meta_payload = {
        "ordermin": ordermin,
        "costmin": costmin,
        "pair_decimals": price_decimals,
        "lot_decimals": lot_decimals,
    }
    _PAIR_CACHE[normalized] = (now, meta_payload)
    return dict(meta_payload)


def kraken_place_order(
    pair: str,
    side: str,
    size: float,
    price: Optional[float] = None,
    *,
    ordertype: Optional[str] = None,
    time_in_force: Optional[str] = None,
    validate: bool = False,
    min_cost_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Submit a spot order to Kraken and return a structured response."""

    if size <= 0:
        raise ValueError("Order size must be positive")
    order_side = side.lower()
    if order_side not in {"buy", "sell"}:
        raise ValueError("Order side must be 'buy' or 'sell'")

    kpair = _normalize_pair(pair)
    resolved_ordertype = ordertype or ("market" if price is None else "limit")

    payload: Dict[str, Any] = {
        "pair": kpair,
        "type": order_side,
        "ordertype": resolved_ordertype,
        "volume": f"{size:.10f}".rstrip("0").rstrip("."),
    }

    attempted_cost: Optional[float] = None
    if price is not None:
        try:
            attempted_cost = round(float(price) * float(size), 8)
        except (TypeError, ValueError):  # defensive only
            attempted_cost = None

    threshold_value = float(min_cost_threshold) if min_cost_threshold is not None else None

    def _auth_failure(message: str) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "ok": False,
            "code": "auth",
            "error": message,
            "pair": pair,
            "side": order_side,
            "endpoint": "AddOrder",
            "txid": None,
            "descr": None,
            "attempted_cost": attempted_cost,
            "threshold": threshold_value,
            "raw": None,
            "errors": None,
        }
        return response

    try:
        _, candidate_secret = _get_credentials()
    except KrakenAuthError as exc:
        logger.error(
            "Kraken order credential error | pair=%s side=%s error=%s",
            pair,
            order_side,
            exc,
        )
        return _auth_failure(str(exc))

    try:
        _decode_secret(candidate_secret)
    except KrakenAuthError:
        logger.error(
            "Kraken order secret decode failed | pair=%s side=%s",
            pair,
            order_side,
        )
        return _auth_failure("Invalid Kraken API secret / malformed base64")

    if resolved_ordertype != "market":
        if price is None:
            raise ValueError("Price is required for non-market orders")
        payload["price"] = f"{price:.10f}".rstrip("0").rstrip(".")

    if time_in_force:
        payload["timeinforce"] = time_in_force
    if validate:
        payload["validate"] = True

    logger.info(
        "Kraken order submit | pair=%s side=%s type=%s volume=%s price=%s validate=%s",
        payload["pair"],
        order_side,
        payload["ordertype"],
        payload["volume"],
        payload.get("price"),
        validate,
    )

    base = _private_request("AddOrder", payload)
    response = _standard_response(
        base,
        endpoint="AddOrder",
        extra={"pair": pair, "side": order_side},
    )

    if not response.get("ok"):
        logger.error(
            "Kraken order error | pair=%s side=%s error=%s",
            pair,
            order_side,
            response.get("error"),
        )
        response.setdefault("result", None)
        response.setdefault("txid", None)
        response.setdefault("descr", None)
        response["pair"] = pair
        response["side"] = order_side
        if attempted_cost is not None:
            response.setdefault("attempted_cost", attempted_cost)
        response.setdefault("threshold", threshold_value)
        if response.get("code") == "cost_minimum_not_met" and attempted_cost is not None:
            logger.warning(
                "Kraken cost minimum not met | pair=%s side=%s attempted_cost=%.8f",
                pair,
                order_side,
                attempted_cost,
            )
        return response

    result = response.get("result") or {}
    descr = result.get("descr") if isinstance(result, dict) else None
    order_descr = descr.get("order") if isinstance(descr, dict) else None
    txid = result.get("txid") if isinstance(result, dict) else None

    logger.info(
        "Kraken order response | pair=%s descript=%s txid=%s raw=%s",
        payload["pair"],
        order_descr,
        txid,
        json.dumps(response.get("raw"), default=str),
    )
    response["txid"] = txid
    response["descr"] = order_descr
    response.setdefault("attempted_cost", attempted_cost)
    response.setdefault("threshold", threshold_value)
    response["pair"] = pair
    response["side"] = order_side
    response["error"] = None
    response["code"] = "ok"
    return response


def kraken_cancel_order(txid: str | Iterable[str]) -> Dict[str, Any]:
    """Cancel one or more orders; returns Kraken's cancellation result."""

    if isinstance(txid, (list, tuple, set)):
        txid_value = ",".join(str(t) for t in txid)
    else:
        txid_value = str(txid)
    logger.info("Kraken cancel order | txid=%s", txid_value)
    response = _private_request("CancelOrder", {"txid": txid_value})
    if response.get("ok"):
        logger.info(
            "Kraken cancel result | txid=%s result=%s",
            txid_value,
            json.dumps(response.get("raw"), default=str),
        )
    else:
        logger.error(
            "Kraken cancel error | txid=%s error=%s",
            txid_value,
            response.get("error"),
        )
    return response


def kraken_open_orders(include_trades: bool = False) -> Dict[str, Any]:
    """Fetch open orders; returns Kraken's raw open-orders payload."""

    payload = {"trades": bool(include_trades)}
    response = _private_request("OpenOrders", payload)
    if response.get("ok"):
        logger.debug(
            "Kraken open orders | trades=%s result=%s",
            include_trades,
            json.dumps(response.get("raw"), default=str),
        )
    else:
        logger.error(
            "Kraken open orders error | trades=%s error=%s",
            include_trades,
            response.get("error"),
        )
    return response


__all__ = [
    "KrakenAPIError",
    "KrakenAuthError",
    "kraken_get_balance",
    "kraken_get_asset_pair_meta",
    "kraken_place_order",
    "kraken_cancel_order",
    "kraken_open_orders",
    "query_api_key_permissions",
]
