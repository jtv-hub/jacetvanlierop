"""Utility script to verify Kraken API credentials and order submission.

- Validates that credentials load correctly from environment/.env.
- Fetches balance using the private API and prints responses.
- Submits a validate-only order to ensure signing and headers are correct.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict
from urllib.parse import urlencode

from crypto_trading_bot.config import (
    CONFIG,
    ConfigurationError,
    _sanitize_base64_secret,
    set_live_mode,
)
from crypto_trading_bot.utils import kraken_client
from crypto_trading_bot.utils.kraken_client import KrakenAPIError

logger = logging.getLogger("verify_kraken")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _mask(value: str) -> str:
    return (value[:6] + "***") if value and len(value) >= 6 else "***"


def _build_and_log_headers(
    endpoint: str,
    payload: Dict[str, Any],
    api_key: str,
    secret: str,
) -> tuple[str, str, Dict[str, str]]:
    nonce = str(int(time.time() * 1000))
    body = dict(payload)
    body["nonce"] = nonce
    postdata = urlencode(body, doseq=True)
    api_sign = kraken_client._sign_request(f"/0/private/{endpoint}", nonce, postdata, secret)
    headers = kraken_client._build_headers(api_key, api_sign)
    logged_headers = {k: (v if k != "API-Key" else _mask(v)) for k, v in headers.items()}
    logger.info("Headers for %s: %s", endpoint, json.dumps(logged_headers, indent=2))
    return nonce, postdata, headers


def _post(endpoint: str, postdata: str, headers: Dict[str, str]) -> Dict[str, Any]:
    url = f"https://api.kraken.com/0/private/{endpoint}"
    try:
        response = kraken_client._http_post(url, postdata, headers, timeout=15.0)
        logger.info("Raw response for %s: %s", endpoint, json.dumps(response, indent=2))
        return response
    except KrakenAPIError as exc:
        logger.error("HTTP error for %s: %s", endpoint, exc)
        raise


def verify() -> int:
    try:
        set_live_mode(True)
        logger.info("Live mode validation succeeded.")
    except ConfigurationError as exc:
        logger.error("Live mode validation failed: %s", exc)
        return 1

    try:
        api_key, raw_secret = kraken_client._get_credentials()
    except KrakenAPIError as exc:
        logger.error("Failed to load credentials: %s", exc)
        set_live_mode(False)
        return 1

    sanitized_secret = _sanitize_base64_secret(raw_secret)
    logger.info(
        "Loaded key_prefix=%s key_origin=%s secret_origin=%s sanitized_secret_length=%d",
        _mask(api_key),
        CONFIG.get("_kraken_key_origin", "unknown"),
        CONFIG.get("_kraken_secret_origin", "unknown"),
        len(sanitized_secret),
    )

    balance_payload: Dict[str, Any] = {}
    try:
        nonce, postdata, headers = _build_and_log_headers("Balance", balance_payload, api_key, sanitized_secret)
        logger.debug("Balance nonce=%s", nonce)
        _post("Balance", postdata, headers)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Balance verification failed: %s", exc)
    else:
        logger.info("Balance verification request completed.")

    order_payload = {
        "pair": "XBTUSD",
        "type": "buy",
        "ordertype": "limit",
        "price": "1",
        "volume": "0.0001",
        "validate": True,
    }
    try:
        nonce, postdata, headers = _build_and_log_headers("AddOrder", order_payload, api_key, sanitized_secret)
        logger.debug("Order nonce=%s", nonce)
        _post("AddOrder", postdata, headers)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Order validation failed: %s", exc)
        set_live_mode(False)
        return 1

    set_live_mode(False)
    logger.info("Kraken credentials appear active and usable.")
    return 0


if __name__ == "__main__":
    sys.exit(verify())
