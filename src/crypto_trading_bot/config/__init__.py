"""Configuration loader for crypto_trading_bot.

Loads environment variables from a `.env` file (project root) and exposes
application configuration via the ``CONFIG`` dictionary.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from typing import Tuple

from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger(__name__)

LIVE_MODE_LABEL = "\U0001f6a8 LIVE MODE \U0001f6a8"
PAPER_MODE_LABEL = "PAPER MODE"
is_live: bool = False
CONFIG: dict = {}


class ConfigurationError(RuntimeError):
    """Raised when mandatory configuration is missing or invalid."""


def _sanitize_value(value: str) -> str:
    """Trim whitespace, strip wrapping quotes, and drop non-printable characters."""

    raw = value or ""
    stripped = raw.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        stripped = stripped[1:-1]
    printable = "".join(ch for ch in stripped if ch.isprintable())
    return printable.strip()


def _read_env_trimmed(name: str) -> str:
    """Return sanitized environment variable ``name``."""

    return _sanitize_value(os.getenv(name) or "")


def _sanitize_base64_secret(secret: str) -> str:
    """Sanitize and pad base64 secrets while preserving only valid characters."""

    cleaned = _sanitize_value(secret)
    cleaned = re.sub(r"[^A-Za-z0-9+/=]", "", cleaned)
    padding = (-len(cleaned)) % 4
    if padding:
        cleaned += "=" * padding
    return cleaned


def _validate_credentials() -> Tuple[str, str]:
    """Ensure Kraken key/secret are present and secret is valid base64."""

    load_dotenv(override=False)
    file_values = dotenv_values() or {}

    file_key = _sanitize_value(file_values.get("KRAKEN_API_KEY", ""))
    file_secret = _sanitize_value(file_values.get("KRAKEN_API_SECRET", ""))

    env_key = _read_env_trimmed("KRAKEN_API_KEY")
    env_secret = _read_env_trimmed("KRAKEN_API_SECRET")

    key_source = "env" if env_key else "dotenv" if file_key else "config"
    secret_source = "env" if env_secret else "dotenv" if file_secret else "config"

    key = env_key or file_key or _sanitize_value(CONFIG.get("kraken_api_key") or "")
    secret_raw = env_secret or file_secret or _sanitize_value(CONFIG.get("kraken_api_secret") or "")

    if not key:
        raise ConfigurationError("Kraken API key must be provided for live mode.")

    secret = _sanitize_base64_secret(secret_raw)

    if not secret:
        raise ConfigurationError("Kraken API secret must be provided for live mode.")

    try:
        base64.b64decode(secret, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.error("Kraken API secret failed base64 validation: %s", exc)
        raise ConfigurationError("Kraken API secret is not valid base64.") from exc

    CONFIG["_kraken_key_origin"] = key_source
    CONFIG["_kraken_secret_origin"] = secret_source
    CONFIG["kraken_api_key"] = key
    CONFIG["kraken_api_secret"] = secret

    key_preview = (key[:6] + "***") if len(key) >= 6 else "***"
    secret_preview = (secret[:6] + "***") if len(secret) >= 6 else "***"

    if len(key) != 56:
        logger.warning("Kraken API key length unexpected: %d", len(key))
    if len(secret) != 88:
        logger.warning("Kraken API secret length unexpected: %d", len(secret))

    logger.debug(
        "Kraken credentials validated | key_prefix=%s key_length=%d secret_prefix=%s secret_length=%d key_source=%s secret_source=%s",
        key_preview,
        len(key),
        secret_preview,
        len(secret),
        key_source,
        secret_source,
    )
    return key, secret


def set_live_mode(flag: bool) -> None:
    """Set the global live-trading toggle with credential validation."""

    global is_live  # noqa: PLW0603 - intentional shared toggle
    if flag:
        _validate_credentials()
    is_live = bool(flag)


def get_mode_label() -> str:
    """Return a human-readable label for the current trading mode."""

    return LIVE_MODE_LABEL if is_live else PAPER_MODE_LABEL


def _to_float(value: str | None, default: float) -> float:
    """Best-effort float conversion with a fallback default."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str | None, default: int) -> int:
    """Best-effort int conversion with a fallback default."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    """Best-effort boolean conversion with a fallback default."""

    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


# Load environment variables from a .env file in the project root
load_dotenv()

CONFIG: dict = {
    "tradable_pairs": [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "LINK/USD",
    ],
    "rsi": {
        "period": 14,
        "lower": 48,
        "upper": 75,
        "exit_upper": 55,
    },
    "max_portfolio_risk": 0.10,
    "min_volume": 100,
    "trade_size": {"min": 0.001, "max": 0.005},
    "slippage": {
        "majors": 0.001,
        "alts_min": 0.005,
        "alts_max": 0.01,
        "use_random": False,
    },
    "buffer_defaults": {
        "trending": 1.0,
        "chop": 0.5,
        "volatile": 0.5,
        "flat": 0.25,
        "unknown": 0.25,
    },
    "correlation": {"window": 30, "threshold": 0.7},
    "paper_mode": {
        "starting_balance": _to_float(
            os.getenv("PAPER_STARTING_BALANCE"),
            100_000.0,
        )
    },
    "live_mode": {
        "balance_env_var": os.getenv(
            "CRYPTO_TRADING_BOT_BALANCE_ENV",
            "CRYPTO_TRADING_BOT_LIVE_BALANCE",
        ),
        "fallback_balance": _to_float(
            os.getenv("LIVE_BALANCE_FALLBACK"),
            0.0,
        ),
        "balance_provider": os.getenv("ACCOUNT_BALANCE_PROVIDER"),
    },
    "auto_pause": {
        "max_drawdown_pct": _to_float(
            os.getenv("AUTO_PAUSE_MAX_DRAWDOWN"),
            0.10,
        ),
        "max_consecutive_losses": _to_int(
            os.getenv("AUTO_PAUSE_MAX_CONSEC_LOSSES"),
            5,
        ),
        "max_total_roi_pct": _to_float(
            os.getenv("AUTO_PAUSE_MAX_TOTAL_ROI"),
            -0.15,
        ),
    },
    "kraken_api_key": _read_env_trimmed("KRAKEN_API_KEY"),
    "kraken_api_secret": _read_env_trimmed("KRAKEN_API_SECRET"),
    "kraken": {
        "balance_asset": (os.getenv("KRAKEN_BALANCE_ASSET") or "USDC").upper(),
        "validate_orders": _to_bool(os.getenv("KRAKEN_VALIDATE_ORDERS"), False),
        "time_in_force": os.getenv("KRAKEN_TIME_IN_FORCE"),
    },
}

if logger.isEnabledFor(logging.DEBUG):
    logger.debug(
        "Config: kraken_api_key_present=%s kraken_api_secret_present=%s",
        bool(CONFIG.get("kraken_api_key")),
        bool(CONFIG.get("kraken_api_secret")),
    )

_env_live_flag = os.getenv("CRYPTO_TRADING_BOT_LIVE")
if _env_live_flag is not None:
    normalized = _env_live_flag.strip().lower()
    set_live_mode(normalized in {"1", "true", "yes", "on"})

    if logger.isEnabledFor(logging.INFO):
        logger.info("Live trading mode overridden by env: %s -> %s", _env_live_flag, is_live)

if is_live:
    _validate_credentials()


__all__ = [
    "CONFIG",
    "LIVE_MODE_LABEL",
    "PAPER_MODE_LABEL",
    "get_mode_label",
    "is_live",
    "set_live_mode",
    "ConfigurationError",
]
