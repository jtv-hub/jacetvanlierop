"""
Shared helper primitives for interacting with Kraken's REST API.

This module exposes low-level utilities used by ``kraken_client`` and other
components that need direct access to credential handling, request signing,
or cached metadata. Higher level business logic should continue to import
the public interface from ``kraken_client``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import urlencode

import requests

from crypto_trading_bot.utils.helpers import (
    build_request_headers,
    fallback_pair_meta,
    offline_private_result,
    redact_nonce,
    resolve_http_client,
)
from crypto_trading_bot.utils.kraken_client_config import CONFIG
from crypto_trading_bot.utils.kraken_client_config import (
    ensure_config_loaded as _ensure_config_loaded,
)
from crypto_trading_bot.utils.kraken_client_config import (
    sanitize_base64_secret as _sanitize_base64_secret,
)
from crypto_trading_bot.utils.kraken_pairs import normalize_pair

logger = logging.getLogger(__name__)

_API_BASE = "https://api.kraken.com"
_PRIVATE_PREFIX = "/0/private/"
_PUBLIC_PREFIX = "/0/public/"
_USER_AGENT = "crypto-trading-bot/1.0 (Kraken private client)"

_ASSET_HINTS: Dict[str, Iterable[str]] = {
    "USDC": ("USDC", "ZUSD", "USDT"),
    "USD": ("ZUSD", "USD"),
    "BTC": ("XXBT", "XBT", "BTC"),
    "ETH": ("XETH", "ETH"),
    "SOL": ("SOL",),
    "LINK": ("LINK",),
    "XRP": ("XXRP", "XRP"),
}

_PAIR_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_PAIR_CACHE_TTL = 300.0
_OFFLINE_ERROR_TOKENS = ("resolve", "dns", "temporary failure")
_OFFLINE_BALANCE_DEFAULT = {"USDC": "1000.30000000", "ZUSD": "1000.30000000"}
_OFFLINE_RIGHTS_DEFAULT = {"rights": {"can_trade": True, "can_withdraw": False}}

sanitize_base64_secret = _sanitize_base64_secret


class KrakenAPIError(RuntimeError):
    """Raised when Kraken returns an error payload or transport fails."""


class KrakenAuthError(KrakenAPIError):
    """Raised when credentials are missing or invalid."""


def _invalidate_pair_cache() -> None:
    """Clear the cached Kraken pair metadata."""

    _PAIR_CACHE.clear()


def _auth_error_response(endpoint: str, message: str) -> Dict[str, Any]:
    """Return a normalized error payload for authentication failures."""

    return {
        "ok": False,
        "error": message,
        "errors": [message],
        "code": "auth",
        "endpoint": endpoint,
        "result": None,
        "raw": None,
    }


def _get_credentials() -> tuple[str, str]:
    """Fetch Kraken API credentials from environment or config."""

    _ensure_config_loaded()

    env_key = os.getenv("KRAKEN_API_KEY", "")
    env_secret = os.getenv("KRAKEN_API_SECRET", "")

    key = (env_key or CONFIG.get("kraken_api_key") or "").strip()
    secret = (env_secret or CONFIG.get("kraken_api_secret") or "").strip()
    if not key or not secret:
        message = "EAuth:Missing credentials"
        logger.error(message)
        raise KrakenAuthError(message)
    key_origin = CONFIG.get("_kraken_key_origin", "unknown")
    secret_origin = CONFIG.get("_kraken_secret_origin", "unknown")
    sanitized_message = " ".join(
        [
            "Kraken credentials sanitised | key_prefix=%s key_length=%d key_origin=%s",
            "secret_length=%d secret_origin=%s",
        ]
    )
    logger.debug(
        sanitized_message,
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
        message = "EAuth:Invalid secret"
        logger.error("Kraken secret sanitization failed: %s", exc)
        raise KrakenAuthError(message) from exc

    if not normalized:
        message = "EAuth:Invalid secret"
        logger.error("Kraken secret missing or empty")
        raise KrakenAuthError(message)

    try:
        decoded = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        message = "EAuth:Invalid secret"
        logger.error("Kraken secret decode failed: %s", exc)
        raise KrakenAuthError(message) from exc

    logger.debug("Kraken secret decoded length=%d", len(decoded))
    return decoded


def _normalize_pair(pair: str) -> str:
    """Return Kraken's altname for a pair like ``BTC/USDC``."""

    return normalize_pair(pair)


def _http_post(url: str, data: str, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    """Compatibility wrapper that raises ``KrakenAPIError`` on transport failure."""

    try:
        response = requests.post(
            url,
            data=data,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:  # pragma: no cover - network dependent
        raise KrakenAPIError(f"HTTP POST failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise KrakenAPIError(f"Invalid JSON response: {exc}") from exc


def _sign_request(url_path: str, nonce: str, postdata: str, secret: str) -> str:
    """Return Kraken HMAC signature for a private request."""

    secret_bytes = _decode_secret(secret)
    sha = hashlib.sha256((nonce + postdata).encode()).digest()
    message = url_path.encode() + sha
    mac = hmac.new(secret_bytes, message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def _offline_private_result(
    endpoint: str,
    *,
    last_error: str | None,
) -> Dict[str, Any] | None:
    """Return a deterministic offline response when Kraken is unreachable."""

    _ensure_config_loaded()
    return offline_private_result(
        endpoint,
        last_error,
        CONFIG,
        error_tokens=_OFFLINE_ERROR_TOKENS,
        balance_defaults=_OFFLINE_BALANCE_DEFAULT,
        rights_defaults=_OFFLINE_RIGHTS_DEFAULT,
    )


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

    try:
        backend_name, client_module = resolve_http_client()
    except RuntimeError as exc:
        raise KrakenAPIError(str(exc)) from exc

    if backend_name == "requests":
        try:
            response = client_module.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (  # type: ignore[attr-defined]
            client_module.exceptions.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            raise KrakenAPIError(f"HTTP GET failed: {exc}") from exc

    if backend_name == "httpx":
        client_class = client_module.Client  # type: ignore[attr-defined]
        http_error = client_module.HTTPError  # type: ignore[attr-defined]
        try:
            with client_class(timeout=timeout, headers=headers) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except (http_error, ValueError, TypeError) as exc:
            raise KrakenAPIError(f"HTTP GET failed: {exc}") from exc

    raise KrakenAPIError("Unsupported HTTP backend configured.")


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
        return _auth_error_response(endpoint, str(exc))

    try:
        secret_bytes = _decode_secret(api_secret)
    except KrakenAuthError as exc:
        logger.error("Kraken secret validation error: %s", exc)
        return _auth_error_response(endpoint, str(exc))

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

        headers = build_request_headers(api_key, api_sign, user_agent=_USER_AGENT)

        try:
            logger.debug(
                "Kraken POST %s data=%s attempt=%d secret_len=%d",
                url_path,
                redact_nonce(postdata),
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
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
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
            if any("eauth" in err for err in normalized_errors):
                logger.error("Kraken auth error payload on %s: %s", endpoint, error_list)
                raise KrakenAuthError(combined_error)
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

    offline_response = _offline_private_result(endpoint, last_error=last_error)
    if offline_response is not None:
        logger.warning(
            "Using offline Kraken stub for %s due to network error: %s",
            endpoint,
            last_error,
        )
        return offline_response

    return {
        "ok": False,
        "error": last_error or "Max attempts exceeded for Kraken request",
        "code": "http",
        "endpoint": endpoint,
        "result": None,
        "raw": None,
    }


def _fallback_pair_meta(pair: str) -> Dict[str, Any]:
    """Return cached or fallback pair metadata."""

    _ensure_config_loaded()
    return fallback_pair_meta(pair, CONFIG, logger)


def _safe_query_call(
    query_func: Optional[Callable[[], Dict[str, Any]]],
    *,
    expected_error: type[BaseException],
    fallback_reason: str = "not callable",
) -> tuple[Dict[str, Any] | None, str | None]:
    """Safely execute a dynamic query callback, returning a tuple of result/error."""

    if not callable(query_func):
        message = f"unavailable ({fallback_reason})"
        logger.warning("query_func is not callable: %s", query_func)
        return None, message

    try:
        result = query_func()
        return (result if isinstance(result, dict) else {}), None
    except (
        expected_error,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:  # pragma: no cover - defensive logging
        message = f"unavailable ({exc})"
        logger.warning("query_func execution failed: %s", exc)
        return None, message


__all__ = [
    "KrakenAPIError",
    "KrakenAuthError",
    "sanitize_base64_secret",
    "_API_BASE",
    "_PRIVATE_PREFIX",
    "_PUBLIC_PREFIX",
    "_USER_AGENT",
    "_ASSET_HINTS",
    "_PAIR_CACHE",
    "_PAIR_CACHE_TTL",
    "_invalidate_pair_cache",
    "_auth_error_response",
    "_get_credentials",
    "_decode_secret",
    "_normalize_pair",
    "_http_post",
    "_http_public",
    "_sign_request",
    "_private_request",
    "_fallback_pair_meta",
    "_safe_query_call",
    "_OFFLINE_ERROR_TOKENS",
    "_OFFLINE_BALANCE_DEFAULT",
    "_OFFLINE_RIGHTS_DEFAULT",
]
