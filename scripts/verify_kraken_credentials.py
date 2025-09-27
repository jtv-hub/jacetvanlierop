# pylint: disable=protected-access

"""Utility script to verify Kraken API credentials and order submission.

- Validates that credentials load correctly from environment/.env.
- Fetches balance using the private API and prints responses.
- Submits a validate-only order to ensure signing and headers are correct.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlencode

from crypto_trading_bot.config import CONFIG, ConfigurationError, set_live_mode
from crypto_trading_bot.utils import kraken_client
from crypto_trading_bot.utils.kraken_client import KrakenAPIError

logger = logging.getLogger("verify_kraken")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _mask(value: str) -> str:
    """Return a masked preview of the given credential string."""

    return (value[:6] + "***") if value and len(value) >= 6 else "***"


def _build_private_request(
    endpoint: str,
    payload: Dict[str, Any],
    api_key: str,
    api_secret: str,
) -> tuple[str, str, Dict[str, str]]:
    """Return (nonce, postdata, headers) for a private Kraken request."""

    nonce = str(int(time.time() * 1000))
    body = dict(payload)
    body["nonce"] = nonce
    postdata = urlencode(body, doseq=True)
    api_sign = kraken_client._sign_request(f"/0/private/{endpoint}", nonce, postdata, api_secret)
    headers = kraken_client._build_headers(api_key, api_sign)
    return nonce, postdata, headers


def _log_private_request(endpoint: str, headers: Dict[str, str]) -> None:
    masked_headers = {k: (v if k != "API-Key" else _mask(v)) for k, v in headers.items()}
    logger.info("Headers for %s: %s", endpoint, json.dumps(masked_headers, indent=2))


def _post(endpoint: str, postdata: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """Execute a private Kraken POST and log the response."""

    base_response = kraken_client._http_post(  # pylint: disable=protected-access
        f"https://api.kraken.com/0/private/{endpoint}",
        postdata,
        headers,
        timeout=15.0,
    )
    logger.info("Raw response for %s: %s", endpoint, json.dumps(base_response, indent=2))
    return base_response


def _ensure_secret_files_present() -> bool:
    """Verify that credential file environment variables are set and readable."""

    key_path = os.getenv("KRAKEN_API_KEY_FILE")
    secret_path = os.getenv("KRAKEN_API_SECRET_FILE")

    missing = False

    if not key_path:
        logger.error(
            "Environment variable %s is not set. Source scripts/set_env.sh before running this script.",
            "KRAKEN_API_KEY_FILE",
        )
        missing = True
    elif not Path(key_path).expanduser().is_file():
        logger.error(
            "KRAKEN_API_KEY_FILE points to %s but the file does not exist. Populate .secrets/kraken_key.txt and rerun.",
            key_path,
        )
        missing = True

    if not secret_path:
        logger.error(
            "Environment variable %s is not set. Source scripts/set_env.sh before running this script.",
            "KRAKEN_API_SECRET_FILE",
        )
        missing = True
    elif not Path(secret_path).expanduser().is_file():
        logger.error(
            "KRAKEN_API_SECRET_FILE points to %s but the file does not exist. "
            "Populate .secrets/kraken_secret.txt and rerun.",
            secret_path,
        )
        missing = True

    if missing:
        logger.info("Populate the credential files, then run: source scripts/set_env.sh")
    return not missing


def verify() -> int:
    """Validate live credentials, balance access, and AddOrder signing."""

    if not _ensure_secret_files_present():
        return 1

    try:
        set_live_mode(True)
        logger.info("Live mode validation succeeded.")
    except ConfigurationError as exc:
        logger.error("Live mode validation failed: %s", exc)
        return 1

    try:
        # pylint: disable-next=protected-access
        api_key, raw_secret = kraken_client._get_credentials()
    except KrakenAPIError as exc:
        logger.error("Failed to load credentials: %s", exc)
        set_live_mode(False)
        return 1

    try:
        # pylint: disable-next=protected-access
        sanitized_secret = kraken_client._sanitize_base64_secret(
            raw_secret,
            strict=True,
        )
    except ValueError as exc:
        logger.error("Secret sanitization failed: %s", exc)
        set_live_mode(False)
        return 1
    logger.info(
        "Loaded key_prefix=%s key_origin=%s secret_origin=%s sanitized_secret_length=%d",
        _mask(api_key),
        CONFIG.get("_kraken_key_origin", "unknown"),
        CONFIG.get("_kraken_secret_origin", "unknown"),
        len(sanitized_secret),
    )

    balance_payload: Dict[str, Any] = {}
    try:
        nonce, postdata, headers = _build_private_request(
            "Balance",
            balance_payload,
            api_key,
            sanitized_secret,
        )
        logger.debug("Balance nonce=%s", nonce)
        _log_private_request("Balance", headers)
        _post("Balance", postdata, headers)
    except (KrakenAPIError, TimeoutError, OSError, ValueError) as exc:
        logger.exception("Balance verification failed: %s", exc)
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
        nonce, postdata, headers = _build_private_request(
            "AddOrder",
            order_payload,
            api_key,
            sanitized_secret,
        )
        logger.debug("Order nonce=%s", nonce)
        _log_private_request("AddOrder", headers)
        _post("AddOrder", postdata, headers)
    except (KrakenAPIError, TimeoutError, OSError, ValueError) as exc:
        logger.exception("Order validation failed: %s", exc)
        set_live_mode(False)
        return 1

    set_live_mode(False)
    logger.info("Kraken credentials appear active and usable.")
    return 0


if __name__ == "__main__":
    sys.exit(verify())
