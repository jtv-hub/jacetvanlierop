"""Configuration loader for crypto_trading_bot.

Loads environment variables from a `.env` file (project root) and exposes
application configuration via the ``CONFIG`` dictionary.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from typing import Tuple

from crypto_trading_bot.utils.secrets_manager import SecretNotFound, get_secret

try:  # pragma: no cover - optional dependency
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - fallback when python-dotenv unavailable

    def dotenv_values(*_args, **_kwargs):
        return {}


try:  # pragma: no cover - handled in tests via monkeypatch
    from crypto_trading_bot.utils.kraken_client import query_api_key_permissions
except Exception:  # pragma: no cover - absence handled safely below
    query_api_key_permissions = None  # type: ignore[assignment]

try:  # pragma: no cover - metadata fetch optional during tests
    from crypto_trading_bot.utils.kraken_client import kraken_get_asset_pair_meta
except Exception:  # pragma: no cover
    kraken_get_asset_pair_meta = None  # type: ignore[assignment]
logger = logging.getLogger(__name__)

LIVE_MODE_LABEL = "\U0001f6a8 LIVE MODE \U0001f6a8"
PAPER_MODE_LABEL = "PAPER MODE"
is_live: bool = False
CONFIG: dict = {}


class ConfigurationError(RuntimeError):
    """Raised when mandatory configuration is missing or invalid."""


def _assert_no_withdraw_rights() -> None:
    if query_api_key_permissions is None:
        logger.warning("Kraken client unavailable; unable to verify withdraw permissions before live mode.")
        return

    try:
        permissions = query_api_key_permissions()
    except Exception as exc:  # pragma: no cover - network issues
        logger.warning("Failed to query Kraken API key permissions: %s", exc)
        return

    if permissions is None:
        logger.warning("QueryKey permissions response was None; proceeding without withdraw check.")
        return

    rights = permissions.get("rights") or {}
    if rights.get("can_withdraw"):
        raise ConfigurationError("Cannot enable live mode: API key has withdraw permissions.")


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


_BASE64_ALLOWED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


def _sanitize_base64_secret(secret: str, *, strict: bool = False) -> str:
    """Sanitize base64 secrets, optionally failing on invalid characters.

    When ``strict`` is ``True`` any character outside the base64 alphabet is
    treated as an error so callers can fail fast instead of attempting to
    "repair" credentials. The non-strict mode keeps the previous behaviour of
    stripping invalid characters (useful for diagnostics) and attempts to pad
    the result so downstream code can emit clearer error messages.
    """

    cleaned = _sanitize_value(secret)
    if not cleaned:
        return ""

    trimmed = cleaned.strip()
    removed_any = False
    # Count how many invalid characters were present at the end so we can
    # optionally add visible padding in non-strict diagnostic mode.
    trailing_invalid = 0
    for ch in reversed(trimmed):
        if ch in _BASE64_ALLOWED:
            break
        trailing_invalid += 1

    sanitized_chars = []
    for ch in trimmed:
        if ch in _BASE64_ALLOWED:
            sanitized_chars.append(ch)
        else:
            removed_any = True

    sanitized = "".join(sanitized_chars)

    if strict and removed_any:
        raise ValueError("Invalid characters detected in Kraken API secret.")

    padding = (-len(sanitized)) % 4
    if padding:
        sanitized += "=" * padding

    if not strict and removed_any and trailing_invalid and not sanitized.endswith("="):
        sanitized += "=" * min(trailing_invalid, 2)

    return sanitized.strip()


def _validate_credentials() -> Tuple[str, str]:
    """Ensure Kraken key/secret are present and secret is valid base64."""

    file_key = ""
    file_secret = ""

    env_key = _read_env_trimmed("KRAKEN_API_KEY")
    env_secret = _read_env_trimmed("KRAKEN_API_SECRET")

    try:
        sm_key = _sanitize_value(get_secret("KRAKEN_API_KEY"))
        key_source = "secrets_manager"
    except SecretNotFound:
        sm_key = ""
        key_source = "env" if env_key else "dotenv" if file_key else "config"

    try:
        sm_secret = _sanitize_value(get_secret("KRAKEN_API_SECRET"))
        secret_source = "secrets_manager"
    except SecretNotFound:
        sm_secret = ""
        secret_source = "env" if env_secret else "dotenv" if file_secret else "config"

    key_config = _sanitize_value(CONFIG.get("kraken_api_key") or "")
    secret_config = _sanitize_value(CONFIG.get("kraken_api_secret") or "")

    key_candidates = [sm_key, env_key, file_key, key_config]
    secret_candidates = [sm_secret, env_secret, file_secret, secret_config]

    key = next((candidate for candidate in key_candidates if candidate), "")
    secret_raw = next((candidate for candidate in secret_candidates if candidate), "")

    if key == sm_key and key:
        key_source = "secrets_manager"
    elif key == env_key and key:
        key_source = "env"
    elif key == file_key and key:
        key_source = "dotenv"
    elif key == key_config and key:
        key_source = "config"

    if secret_raw == sm_secret and secret_raw:
        secret_source = "secrets_manager"
    elif secret_raw == env_secret and secret_raw:
        secret_source = "env"
    elif secret_raw == file_secret and secret_raw:
        secret_source = "dotenv"
    elif secret_raw == secret_config and secret_raw:
        secret_source = "config"

    if not key:
        message = "Kraken API key/secret missing for live mode."
        logger.error(message)
        raise ConfigurationError(message)

    try:
        secret = _sanitize_base64_secret(secret_raw, strict=True)
    except ValueError as exc:
        logger.error(
            "Kraken API secret is not valid base64 (failed base64 sanitization): %s",
            exc,
        )
        raise ConfigurationError("Kraken API secret is not valid base64.") from exc

    if not secret:
        message = "Kraken API secret must be provided for live mode."
        logger.error(message)
        raise ConfigurationError(message)

    try:
        base64.b64decode(secret, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.error(
            "Kraken API secret is not valid base64 (failed base64 decode): %s",
            exc,
        )
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
        "Kraken credentials validated | key_prefix=%s key_length=%d "
        "secret_prefix=%s secret_length=%d key_source=%s secret_source=%s",
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

    if flag:
        try:
            _validate_credentials()
        except ConfigurationError as exc:
            logger.error("Kraken API key/secret validation failed: %s", exc)
            raise
        _assert_no_withdraw_rights()
    globals()["is_live"] = bool(flag)


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


def _parse_pair_list(raw: str) -> list[str]:
    pairs: list[str] = []
    for token in raw.split(","):
        cleaned = _sanitize_value(token).upper()
        if cleaned and "/" in cleaned:
            pairs.append(cleaned)
    return pairs


def _load_tradable_pairs() -> list[str]:
    """Return a list of tradable pairs validated against Kraken metadata."""

    env_pairs = _parse_pair_list(os.getenv("CRYPTO_TRADING_BOT_PAIRS", ""))
    default_pairs = [
        "BTC/USDC",
        "ETH/USDC",
        "SOL/USDC",
        "XRP/USDC",
        "LINK/USDC",
    ]

    candidates = env_pairs or default_pairs
    validated: list[str] = []
    rejected: list[str] = []

    if not candidates:
        return []

    for pair in candidates:
        if kraken_get_asset_pair_meta is None:
            logger.warning(
                "Kraken metadata unavailable; accepting pair %s without validation",
                pair,
            )
            validated.append(pair)
            continue

        try:
            meta = kraken_get_asset_pair_meta(pair)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(
                "Failed to validate pair %s via Kraken metadata: %s",
                pair,
                exc,
            )
            rejected.append(pair)
            continue

        if not meta:
            logger.critical("Pair %s rejected: empty Kraken metadata payload", pair)
            rejected.append(pair)
            continue

        quote = str(meta.get("quote", "")).upper()
        altname = str(meta.get("altname", "")).upper()
        if quote == "USDC" or altname.endswith("USDC"):
            validated.append(pair)
        else:
            logger.critical(
                "Pair %s rejected: Kraken metadata quote=%s altname=%s (expected USDC)",
                pair,
                quote,
                altname,
            )
            rejected.append(pair)

    if rejected:
        logger.critical(
            "Rejected USDC pairs: %s",
            ", ".join(sorted(set(rejected))) or "<none>",
        )
    if validated:
        logger.info(
            "Validated USDC pairs: %s",
            ", ".join(sorted(set(validated))) or "<none>",
        )
        return validated

    logger.critical("No tradable pairs validated; falling back to defaults without metadata confirmation.")
    return default_pairs


def _build_trade_size_config(pairs: list[str]) -> dict:
    """Return per-pair trade size constraints derived from Kraken metadata."""

    default_min = float(os.getenv("CRYPTO_TRADING_BOT_DEFAULT_MIN_VOLUME", "0.001"))
    default_max = float(os.getenv("CRYPTO_TRADING_BOT_DEFAULT_MAX_VOLUME", "0.005"))
    per_pair: dict[str, dict[str, float]] = {}

    if not pairs:
        return {
            "default_min": default_min,
            "default_max": default_max,
            "per_pair": per_pair,
        }

    for pair in pairs:
        if kraken_get_asset_pair_meta is None:
            logger.warning("Kraken metadata unavailable; using fallback trade sizes for %s", pair)
            continue
        try:
            meta = kraken_get_asset_pair_meta(pair)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(
                "Failed to load trade size metadata for %s: %s — using defaults",
                pair,
                exc,
            )
            continue
        min_volume = float(meta.get("ordermin", default_min) or default_min)
        min_cost = float(meta.get("costmin", 0.0) or 0.0)
        per_pair[pair] = {
            "min_volume": min_volume,
            "min_cost": min_cost,
        }

    return {
        "default_min": default_min,
        "default_max": default_max,
        "per_pair": per_pair,
    }


_TRADABLE_PAIRS = _load_tradable_pairs()


# Load environment variables from a .env file in the project root
CONFIG: dict = {
    "tradable_pairs": list(_TRADABLE_PAIRS),
    "rsi": {
        "period": 14,
        "lower": 48,
        "upper": 75,
        "exit_upper": 55,
    },
    "max_portfolio_risk": 0.10,
    "min_volume": 100,
    "trade_size": _build_trade_size_config(_TRADABLE_PAIRS),
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
    logger.warning(
        "Environment variable CRYPTO_TRADING_BOT_LIVE is ignored. Use the CLI "
        "--confirm-live-mode flag to enable live trading intentionally."
    )

CONFIG.setdefault("prelaunch_guard", {})
CONFIG["prelaunch_guard"].setdefault(
    "alert_window_hours", int(os.getenv("CRYPTO_TRADING_BOT_ALERT_WINDOW_HOURS", "72"))
)
CONFIG["prelaunch_guard"].setdefault(
    "max_recent_high_severity", int(os.getenv("CRYPTO_TRADING_BOT_MAX_RECENT_HIGH_SEVERITY", "50"))
)

CONFIG["test_mode"] = _to_bool(os.getenv("CRYPTO_TRADING_BOT_TEST_MODE"), False)

_LIVE_MODE_ENV = _to_bool(os.getenv("LIVE_MODE"), False)
CONFIG.setdefault("live_mode", {})["requested_via_env"] = _LIVE_MODE_ENV

if _LIVE_MODE_ENV:
    try:
        set_live_mode(True)
        logger.warning("LIVE_MODE enabled via environment variable.")
    except ConfigurationError as exc:
        logger.critical("Failed to enable LIVE_MODE from environment: %s", exc)
        raise


__all__ = [
    "CONFIG",
    "LIVE_MODE_LABEL",
    "PAPER_MODE_LABEL",
    "get_mode_label",
    "is_live",
    "set_live_mode",
    "ConfigurationError",
]
