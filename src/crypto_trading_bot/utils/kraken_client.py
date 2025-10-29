"""Kraken private REST client helpers.

This module implements minimal authenticated endpoints required for live
trading: balance lookups, order submission, optional cancellation, and
open-order inspection. It relies on API credentials provided via
environment variables loaded in ``crypto_trading_bot.config``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, Optional

import requests

from crypto_trading_bot.utils.helpers import asset_candidates, standard_response
from crypto_trading_bot.utils.kraken_client_config import (
    CONFIG,
    get_config_module,
)
from crypto_trading_bot.utils.kraken_client_config import (
    ensure_config_loaded as _ensure_config_loaded,
)
from crypto_trading_bot.utils.kraken_helpers import (
    _ASSET_HINTS,
    _PAIR_CACHE,
    _PAIR_CACHE_TTL,
    KrakenAPIError,
    KrakenAuthError,
    _decode_secret,
    _fallback_pair_meta,
    _get_credentials,
    _http_public,
    _invalidate_pair_cache,
    _normalize_pair,
    _private_request,
    _safe_query_call,
    sanitize_base64_secret,
)
from crypto_trading_bot.utils.kraken_helpers import (
    _http_post as _kraken_http_post,
)
from crypto_trading_bot.utils.kraken_helpers import (
    _sign_request as _kraken_sign_request,
)

logger = logging.getLogger(__name__)


def get_client(_config: dict | None = None):
    """Return the active Kraken client module.

    This hook allows callers (e.g., configuration validators) to obtain a
    reference to the client while remaining compatible with future injection.
    The optional ``_config`` argument is accepted for signature compatibility
    with potential future call sites.
    """

    return sys.modules[__name__]


def describe_api_permissions() -> str:
    """Return a human-readable summary of current API key permissions."""

    _ensure_config_loaded()
    module = get_config_module()
    if module is None:
        return "unavailable (config import failed)"

    query_func: Optional[Callable[[], Dict[str, Any]]] = getattr(
        module,
        "query_api_key_permissions",
        None,
    )
    kraken_error = getattr(module, "KrakenAPIError", RuntimeError)

    permissions, error_text = _safe_query_call(
        query_func,
        expected_error=kraken_error,
        fallback_reason="missing query_api_key_permissions",
    )
    if permissions is None:
        return error_text or "unavailable"
    rights = permissions.get("rights", {})
    try:
        return json.dumps(rights, sort_keys=True)
    except TypeError:  # pragma: no cover - defensive
        return str(rights)


def query_private(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 15.0,
    max_attempts: int = 2,
    raise_for_error: bool = True,
) -> Dict[str, Any]:
    """Public helper for Kraken authenticated REST endpoints.

    Returns the normalized response payload produced by ``_private_request``.
    When ``raise_for_error`` is ``True`` (default), a ``KrakenAPIError`` is
    raised if Kraken returns an error payload. Callers that want to inspect
    the error details manually can set ``raise_for_error=False``.
    """

    response = _private_request(
        method,
        dict(params or {}),
        timeout=timeout,
        max_attempts=max_attempts,
    )

    if raise_for_error and not response.get("ok"):
        error_text = response.get("error") or response.get("code") or "unknown error"
        raise KrakenAPIError(f"Kraken {method} failed: {error_text}")

    return response


def kraken_get_balance(asset: str = "USDC") -> Dict[str, Any]:
    """Return the available balance for ``asset`` (defaults to USDC)."""

    base = query_private("Balance", raise_for_error=False)
    response = standard_response(base, endpoint="Balance", extra={"asset": asset.upper()})
    if not response.get("ok"):
        response.setdefault("balance", None)
        return response

    candidates = asset_candidates(asset, _ASSET_HINTS)
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


def get_usdc_balance() -> float | None:
    """Return the current USDC balance as a float, or ``None`` on failure."""

    response = kraken_get_balance("USDC")
    if not isinstance(response, dict):
        logger.warning("USDC balance fetch returned non-dict response: %r", response)
        return None

    balance = response.get("balance")
    if balance is None:
        result_block = response.get("result") or {}
        for key in ("USDC", "ZUSD", "USD"):
            if result_block.get(key) is not None:
                balance = result_block[key]
                logger.debug("USDC balance fallback key=%s value=%s", key, balance)
                break

    if balance is None:
        logger.warning("USDC balance missing from Kraken response: %s", response)
        return None

    try:
        decimal_value = Decimal(str(balance))
    except (InvalidOperation, ValueError, TypeError):
        logger.warning("USDC balance payload not numeric: %r", balance)
        return None

    if not decimal_value.is_finite():
        logger.warning("USDC balance non-finite: %s", decimal_value)
        return None

    rounded = decimal_value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    logger.debug(
        "Fetched USDC balance raw=%s rounded=%s code=%s",
        balance,
        rounded,
        response.get("code"),
    )
    return float(rounded)


def query_api_key_permissions() -> Optional[Dict[str, Any]]:
    """Return a best-effort view of API permissions without using Kraken's deprecated QueryKey."""

    response = _private_request("Balance", {})
    normalized = standard_response(response, endpoint="Balance")
    if not normalized.get("ok"):
        error_text = normalized.get("error") or ""
        code = (normalized.get("code") or "").lower()
        if code == "auth" or "eauth" in error_text.lower():
            raise KrakenAuthError(error_text or "EAuth:Missing credentials")
        logger.warning(
            "Unable to infer Kraken API key permissions from balance endpoint: %s",
            normalized.get("error") or normalized.get("code"),
        )
        return None

    result = normalized.get("result") or {}
    # Balance succeeds when the key has trade/query rights; withdrawals are never required.
    rights = {
        "can_trade": bool(result),
        "can_withdraw": False,
    }
    return {"rights": rights, "raw": result}


def kraken_get_asset_pair_meta(pair: str) -> Dict[str, Any]:
    """Return Kraken asset pair metadata including order and cost minimums."""

    normalized = _normalize_pair(pair)
    now = time.monotonic()
    cached = _PAIR_CACHE.get(normalized)
    if cached and now - cached[0] < _PAIR_CACHE_TTL:
        return dict(cached[1])

    try:
        payload = _http_public("AssetPairs", {"pair": normalized})
    except KrakenAPIError as exc:
        logger.error("AssetPairs HTTP failure for %s: %s", pair, exc)
        fallback = _fallback_pair_meta(pair)
        _PAIR_CACHE[normalized] = (now, fallback)
        return dict(fallback)

    errors = payload.get("error")
    if errors:
        logger.error("AssetPairs error for %s: %s", pair, errors)
        fallback = _fallback_pair_meta(pair)
        _PAIR_CACHE[normalized] = (now, fallback)
        return dict(fallback)

    result = payload.get("result")
    if not isinstance(result, dict):
        logger.error("Invalid AssetPairs payload for %s: %s", pair, payload)
        fallback = _fallback_pair_meta(pair)
        _PAIR_CACHE[normalized] = (now, fallback)
        return dict(fallback)

    meta = result.get(normalized)
    if not isinstance(meta, dict):
        logger.error("Asset metadata missing for pair %s in payload: %s", pair, result)
        fallback = _fallback_pair_meta(pair)
        _PAIR_CACHE[normalized] = (now, fallback)
        return dict(fallback)

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
        return _auth_failure("EAuth:Invalid secret")

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
    response = standard_response(
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
        response.setdefault("txid", [])
        response.setdefault("txid_list", [])
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
    txid_source = result.get("txid") if isinstance(result, dict) else None
    txids: list[str] = []
    if isinstance(txid_source, (list, tuple)):
        txids = [str(item) for item in txid_source if item]
    elif isinstance(txid_source, str) and txid_source:
        txids = [txid_source]
    txid_value = txids[0] if txids else None

    fills_raw = result.get("fills")
    fills: list[dict[str, float | str]] = []
    gross_total = 0.0
    volume_total = 0.0
    fee_total = 0.0
    if isinstance(fills_raw, list):
        for entry in fills_raw:
            if not isinstance(entry, dict):
                continue
            try:
                price_val = float(entry.get("price", entry.get("avg_price", 0.0)) or 0.0)
            except (TypeError, ValueError):
                price_val = 0.0
            try:
                qty_val = float(entry.get("qty", entry.get("volume", entry.get("vol", 0.0))) or 0.0)
            except (TypeError, ValueError):
                qty_val = 0.0
            try:
                cost_val = float(entry.get("cost", price_val * qty_val) or 0.0)
            except (TypeError, ValueError):
                cost_val = price_val * qty_val
            try:
                fee_val = float(entry.get("fee", 0.0) or 0.0)
            except (TypeError, ValueError):
                fee_val = 0.0
            gross_total += cost_val
            volume_total += qty_val
            fee_total += fee_val
            fills.append(
                {
                    "price": price_val,
                    "quantity": qty_val,
                    "cost": cost_val,
                    "fee": fee_val,
                    "type": entry.get("type") or "",
                    "time": entry.get("time") or "",
                }
            )

    if gross_total <= 0.0:
        gross_total = float(result.get("cost") or attempted_cost or 0.0)
    if volume_total <= 0.0:
        volume_total = float(result.get("vol_executed", size) or size)
    if fee_total <= 0.0:
        fee_total = float(result.get("fee") or 0.0)

    average_price = None
    try:
        if volume_total:
            average_price = gross_total / volume_total
    except (TypeError, ZeroDivisionError):
        average_price = None

    if gross_total == 0.0 and attempted_cost:
        gross_total = float(attempted_cost)
    if average_price is None:
        average_price = float(price) if price is not None else None

    if order_side == "buy":
        net_amount = -(gross_total + fee_total)
    else:
        net_amount = gross_total - fee_total

    logger.info(
        "Kraken order response | pair=%s descript=%s txid=%s gross=%.8f fee=%.8f fills=%d",
        payload["pair"],
        order_descr,
        txid_value,
        gross_total,
        fee_total,
        len(fills),
    )
    # Kraken returns txid as a list; preserve that and also surface the first entry for convenience.
    response["txid"] = txids
    response["txid_list"] = txids
    response["txid_single"] = txid_value
    response["descr"] = order_descr
    response.setdefault("attempted_cost", attempted_cost)
    response.setdefault("threshold", threshold_value)
    response["pair"] = pair
    response["side"] = order_side
    response["fills"] = fills
    response["gross_amount"] = gross_total
    response["fee"] = fee_total
    response["net_amount"] = net_amount
    response["filled_volume"] = volume_total
    response["average_price"] = average_price
    response["balance_delta"] = net_amount
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


class KrakenClient:
    """State-free convenience wrapper for Kraken private endpoints."""

    def query_private(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: float = 15.0,
        max_attempts: int = 2,
        raise_for_error: bool = True,
        return_result: bool = True,
    ) -> Dict[str, Any] | Any:
        """Call a Kraken private endpoint and optionally return only the result."""

        payload = query_private(
            method,
            params=params,
            timeout=timeout,
            max_attempts=max_attempts,
            raise_for_error=raise_for_error,
        )
        if return_result and payload.get("ok"):
            return payload.get("result")
        return payload

    def balance(
        self,
        asset: str = "USDC",
        *,
        raise_for_error: bool = True,
    ) -> Dict[str, Any]:
        """Convenience wrapper around ``kraken_get_balance``."""

        response = kraken_get_balance(asset)
        if raise_for_error and not response.get("ok"):
            error_text = response.get("error") or response.get("code") or "unknown error"
            raise KrakenAPIError(f"Kraken balance lookup failed: {error_text}")
        return response

    def trades_history(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        raise_for_error: bool = True,
        return_result: bool = True,
    ) -> Dict[str, Any] | Any:
        """Thin wrapper for Kraken's ``TradesHistory`` endpoint."""

        return self.query_private(
            "TradesHistory",
            params=params,
            raise_for_error=raise_for_error,
            return_result=return_result,
        )

    def raw_balance(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        raise_for_error: bool = True,
    ) -> Dict[str, Any] | Any:
        """Direct access to the ``Balance`` endpoint without normalization."""

        return self.query_private(
            "Balance",
            params=params,
            raise_for_error=raise_for_error,
            return_result=False,
        )


_ensure_config_loaded()

kraken_client = KrakenClient()


__all__ = (
    "KrakenAPIError",
    "KrakenAuthError",
    "KrakenClient",
    "kraken_client",
    "query_private",
    "kraken_get_balance",
    "kraken_get_asset_pair_meta",
    "kraken_place_order",
    "kraken_cancel_order",
    "kraken_open_orders",
    "query_api_key_permissions",
    "sanitize_base64_secret",
    "_invalidate_pair_cache",
    "CONFIG",
    "requests",
    "_http_post",
    "_sign_request",
)


def _http_post(url: str, data: str, headers: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Compatibility wrapper exposing helper HTTP POST for tests."""

    return _kraken_http_post(url, data, headers, timeout)


def _sign_request(url_path: str, nonce: str, postdata: str, secret: str) -> str:
    """Compatibility wrapper that delegates to the shared signing helper."""

    return _kraken_sign_request(url_path, nonce, postdata, secret)
