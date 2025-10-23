"""Shared utility helpers for HTTP requests and credential sanitization."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

_BASE64_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
_HTTP_BACKEND: Tuple[str, Any] | None = None


def sanitize_base64_secret(secret: str, *, strict: bool = False) -> str:
    """Return a base64-safe string, optionally erroring if invalid chars are found."""

    cleaned = (secret or "").strip()
    if not cleaned:
        return ""

    sanitized_chars: list[str] = []
    invalid_seen = False
    for char in cleaned:
        if char in _BASE64_CHARS:
            sanitized_chars.append(char)
        else:
            invalid_seen = True

    if strict and invalid_seen:
        raise ValueError("Invalid characters detected in base64 secret.")

    sanitized = "".join(sanitized_chars)
    padding = (-len(sanitized)) % 4
    if padding:
        sanitized += "=" * padding
    return sanitized


def resolve_http_client(preferred_backends: Iterable[str] | None = None) -> Tuple[str, Any]:
    """Return a cached HTTP client module (requests or httpx)."""

    global _HTTP_BACKEND  # pylint: disable=global-statement
    if _HTTP_BACKEND is not None:
        return _HTTP_BACKEND

    candidates = tuple(preferred_backends or ("requests", "httpx"))
    for backend_name in candidates:
        try:
            module = __import__(backend_name)  # type: ignore[import]
        except ImportError:  # pragma: no cover - optional dependency
            continue
        _HTTP_BACKEND = (backend_name, module)
        return _HTTP_BACKEND

    raise RuntimeError("Missing HTTP client dependency. Install 'requests' or 'httpx'.")


def build_request_headers(api_key: str, api_sign: str, *, user_agent: str) -> Dict[str, str]:
    """Return Kraken-compatible headers for authenticated requests."""

    return {
        "API-Key": api_key,
        "API-Sign": api_sign,
        "User-Agent": user_agent,
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "Accept": "application/json",
    }


def http_post(url: str, data: str, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    """Perform an HTTP POST using the resolved client backend."""

    backend_name, client_module = resolve_http_client()

    if backend_name == "requests":
        try:
            response = client_module.post(url, data=data, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (  # type: ignore[attr-defined]
            client_module.exceptions.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            raise RuntimeError(f"HTTP POST failed: {exc}") from exc

    if backend_name == "httpx":
        client_class = client_module.Client  # type: ignore[attr-defined]
        http_error = client_module.HTTPError  # type: ignore[attr-defined]
        try:
            with client_class(timeout=timeout, headers=headers) as client:
                response = client.post(url, content=data)
                response.raise_for_status()
                return response.json()
        except (http_error, ValueError, TypeError) as exc:
            raise RuntimeError(f"HTTP POST failed: {exc}") from exc

    raise RuntimeError("Unsupported HTTP backend configured.")


def asset_candidates(asset: str, hints: Dict[str, Iterable[str]]) -> Tuple[str, ...]:
    """Return possible Kraken asset codes for ``asset``."""

    if not asset:
        return ("",)
    symbol = asset.upper()
    hint_values = hints.get(symbol)
    if hint_values:
        return tuple(dict.fromkeys(hint_values))
    candidates = [symbol]
    if not symbol.startswith(("X", "Z")):
        candidates.append(f"X{symbol}")
        candidates.append(f"Z{symbol}")
    return tuple(dict.fromkeys(candidates))


def redact_nonce(postdata: str) -> str:
    """Mask nonce values in query strings."""

    if "nonce=" not in postdata:
        return postdata
    parts = ["nonce=***" if token.startswith("nonce=") else token for token in postdata.split("&")]
    return "&".join(parts)


def fallback_pair_meta(pair: str, config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Return safe trade metadata for ``pair`` using local configuration."""

    trade_cfg = config.get("trade_size", {}) or {}
    per_pair = trade_cfg.get("per_pair", {}) or {}
    pair_cfg = per_pair.get(pair, {}) or {}

    default_min_volume = float(trade_cfg.get("default_min", 0.0) or 0.0)
    fallback_volume = float(pair_cfg.get("min_volume", default_min_volume) or default_min_volume)

    kraken_cfg = config.get("kraken", {}) or {}
    default_cost_threshold = float(
        config.get("kraken_min_cost_threshold", kraken_cfg.get("min_cost_threshold", 0.0)) or 0.0
    )
    fallback_cost = float(pair_cfg.get("min_cost", default_cost_threshold) or default_cost_threshold)

    price_decimals = int(kraken_cfg.get("pair_decimals_default", 5) or 5)
    lot_decimals = int(kraken_cfg.get("lot_decimals_default", 8) or 8)

    payload = {
        "ordermin": max(fallback_volume, 0.0),
        "costmin": max(fallback_cost, 0.0),
        "pair_decimals": max(price_decimals, 0),
        "lot_decimals": max(lot_decimals, 0),
        "source": "fallback",
    }
    logger.warning("Using fallback Kraken metadata for %s: %s", pair, payload)
    return payload


def offline_private_result(
    endpoint: str,
    last_error: str | None,
    config: Dict[str, Any],
    *,
    error_tokens: Iterable[str],
    balance_defaults: Dict[str, str],
    rights_defaults: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build deterministic offline responses for private endpoints."""

    if not last_error:
        return None
    lowered = last_error.lower()
    if not any(token in lowered for token in error_tokens):
        return None

    offline_cfg = config.get("kraken_offline") or config.get("kraken", {}).get("offline", {})

    if endpoint == "Balance":
        balances_cfg = offline_cfg.get("balances") if isinstance(offline_cfg, dict) else None
        balances: Dict[str, str] = {}
        if isinstance(balances_cfg, dict):
            for key, value in balances_cfg.items():
                try:
                    balances[str(key).upper()] = f"{float(value):.8f}"
                except (TypeError, ValueError):
                    continue
        if not balances:
            balances = dict(balance_defaults)
        raw = {"error": [], "result": balances}
        return {
            "ok": True,
            "error": None,
            "code": "offline",
            "endpoint": endpoint,
            "result": balances,
            "raw": raw,
            "offline": True,
        }

    if endpoint == "QueryKey":
        rights_cfg: Dict[str, Any] = {}
        if isinstance(offline_cfg, dict):
            candidate = offline_cfg.get("rights")
            if isinstance(candidate, dict):
                rights_cfg = dict(candidate)
        if not rights_cfg:
            rights_cfg = dict(rights_defaults.get("rights", {}))
        payload = {"rights": rights_cfg}
        raw = {"error": [], "result": payload}
        return {
            "ok": True,
            "error": None,
            "code": "offline",
            "endpoint": endpoint,
            "result": payload,
            "raw": raw,
            "offline": True,
        }

    return None


def standard_response(
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


__all__ = [
    "asset_candidates",
    "build_request_headers",
    "fallback_pair_meta",
    "http_post",
    "offline_private_result",
    "redact_nonce",
    "resolve_http_client",
    "sanitize_base64_secret",
    "standard_response",
]
