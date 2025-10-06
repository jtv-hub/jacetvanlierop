"""Configuration loader for crypto_trading_bot.

Loads environment variables from a `.env` file (project root) and exposes
application configuration via the ``CONFIG`` dictionary.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - fallback when python-dotenv unavailable

    def dotenv_values(*_args, **_kwargs):
        """Return an empty mapping when python-dotenv is not installed."""
        return {}


from crypto_trading_bot.utils.secrets_manager import SecretNotFound, get_secret

logger = logging.getLogger(__name__)

LIVE_MODE_LABEL = "\U0001f6a8 LIVE MODE \U0001f6a8"
PAPER_MODE_LABEL = "PAPER MODE"
is_live: bool = False
CONFIG: dict = {}
_DEFAULT_LIVE_CONFIRMATION_FILE = ".confirm_live_trade"


class ConfigurationError(RuntimeError):
    """Raised when mandatory configuration is missing or invalid."""


try:  # pragma: no cover - handled in tests via monkeypatch
    from crypto_trading_bot.utils import kraken_client as _kraken_client
except ImportError:  # pragma: no cover - optional dependency
    _kraken_client = None

if _kraken_client is not None:
    KrakenAPIError = _kraken_client.KrakenAPIError  # type: ignore[attr-defined]
else:  # pragma: no cover - fallback when client is unavailable

    class KrakenAPIError(RuntimeError):
        """Placeholder exception when Kraken client is not importable."""


_WITHDRAW_WARNING_LOGGED = False


def query_api_key_permissions() -> dict[str, dict[str, bool]]:
    """Return Kraken API key permissions (stubbed when client unavailable)."""

    if _kraken_client is None or not hasattr(_kraken_client, "query_api_key_permissions"):
        return {"rights": {"can_withdraw": False}}

    try:
        permissions = _kraken_client.query_api_key_permissions()
    except (KrakenAPIError, RuntimeError, ValueError) as exc:  # pragma: no cover - network issues
        logger.warning("Failed to query Kraken API key permissions: %s", exc)
        return {"rights": {"can_withdraw": False}}

    if permissions is None:
        logger.warning(
            "QueryKey permissions response was None; proceeding without withdraw check.",
        )
        return {"rights": {"can_withdraw": False}}

    return permissions


def _assert_no_withdraw_rights() -> None:
    """Validate that the configured API key cannot perform withdrawals."""

    if _kraken_client is None:
        global _WITHDRAW_WARNING_LOGGED  # pylint: disable=global-statement
        if not _WITHDRAW_WARNING_LOGGED:
            warning = " ".join(
                [
                    "Kraken client unavailable; unable to verify withdraw permissions",
                    "before live mode.",
                ]
            )
            logger.warning(warning)
            _WITHDRAW_WARNING_LOGGED = True
        return

    permissions = query_api_key_permissions()
    if permissions is None:
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


def _kraken_pair_meta(pair: str) -> Optional[Dict[str, Any]]:
    """Return Kraken asset pair metadata when the client helper is available."""

    if _kraken_client is None:
        return None
    return _kraken_client.kraken_get_asset_pair_meta(pair)


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


def _load_dotenv_into_env() -> Dict[str, str]:
    """Load variables from a project-level .env file into ``os.environ``.

    Existing environment variables are never overwritten. Returns the mapping
    of keys loaded from the file so callers can perform follow-up validation.
    """

    candidates = []
    root_env = os.getenv("CRYPTO_TRADING_BOT_ROOT")
    if root_env:
        candidates.append(Path(root_env).expanduser())
    try:
        candidates.append(Path(__file__).resolve().parents[3])
    except IndexError:  # pragma: no cover - defensive on shallow paths
        candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())

    loaded: Dict[str, str] = {}
    for base in candidates:
        env_path = base / ".env"
        if not env_path.is_file():
            continue
        try:
            raw_values = dotenv_values(env_path)
        except (OSError, ValueError):  # pragma: no cover - dotenv internals
            continue
        for key, value in (raw_values or {}).items():
            if value is None or key in loaded:
                continue
            string_value = str(value)
            loaded[key] = string_value
            if not os.getenv(key):
                os.environ[key] = string_value
        break
    if loaded and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Loaded %d entries from .env (without overriding existing environment)",
            len(loaded),
        )
    return loaded


def _read_secret_file(path: str) -> str:
    """Read and sanitize a secret from ``path`` returning an empty string on failure."""

    try:
        content = Path(path).expanduser().read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.error("Unable to read Kraken credential file %s: %s", path, exc)
        return ""
    return _sanitize_value(content)


def _resolve_env_or_file(name: str, file_env: str) -> tuple[str, str]:
    """Return (value, origin) preferring direct environment over credential file."""

    env_value = _sanitize_value(os.getenv(name) or "")
    if env_value:
        return env_value, "env"

    file_path = _sanitize_value(os.getenv(file_env) or "")
    if not file_path:
        return "", "missing"

    file_value = _read_secret_file(file_path)
    if file_value:
        os.environ.setdefault(name, file_value)
        return file_value, f"file:{file_path}"

    logger.error("Credential file %s specified in %s could not be read.", file_path, file_env)
    return "", f"file_error:{file_path}"


def _ensure_env_credentials() -> None:
    """Ensure Kraken credentials are present in ``os.environ`` after dotenv load."""

    api_key, key_origin = _resolve_env_or_file("KRAKEN_API_KEY", "KRAKEN_API_KEY_FILE")
    api_secret, secret_origin = _resolve_env_or_file("KRAKEN_API_SECRET", "KRAKEN_API_SECRET_FILE")

    if api_key:
        os.environ.setdefault("KRAKEN_API_KEY", api_key)
        if key_origin == "env":
            logger.info("Kraken API key sourced from environment variable.")
        elif key_origin.startswith("file:"):
            logger.info(
                "Kraken API key loaded from credential file %s",
                key_origin.split(":", 1)[1],
            )
    if api_secret:
        os.environ.setdefault("KRAKEN_API_SECRET", api_secret)
        if secret_origin == "env":
            logger.info("Kraken API secret sourced from environment variable.")
        elif secret_origin.startswith("file:"):
            logger.info(
                "Kraken API secret loaded from credential file %s",
                secret_origin.split(":", 1)[1],
            )

    if not api_key or not api_secret:
        message = " ".join(
            [
                "Kraken API credentials missing:",
                "ensure KRAKEN_API_KEY and KRAKEN_API_SECRET are set",
            ]
        )
        logger.critical(message)
        raise ConfigurationError(message)

    logger.debug(
        "Kraken API credentials available (sources: %s/%s)",
        key_origin,
        secret_origin,
    )


def _validate_credentials() -> Tuple[str, str]:
    """Ensure Kraken key/secret are present and secret is valid base64."""

    direct_key, direct_key_origin = _resolve_env_or_file(
        "KRAKEN_API_KEY",
        "KRAKEN_API_KEY_FILE",
    )
    direct_secret, direct_secret_origin = _resolve_env_or_file(
        "KRAKEN_API_SECRET",
        "KRAKEN_API_SECRET_FILE",
    )

    key = _sanitize_value(direct_key)
    secret_raw = _sanitize_value(direct_secret)
    key_source = ""
    if direct_key_origin == "env":
        key_source = "env"
    elif direct_key_origin.startswith("file:"):
        key_source = "file"
    secret_source = ""
    if direct_secret_origin == "env":
        secret_source = "env"
    elif direct_secret_origin.startswith("file:"):
        secret_source = "file"

    try:
        sm_key = _sanitize_value(get_secret("KRAKEN_API_KEY"))
    except SecretNotFound:
        sm_key = ""
    try:
        sm_secret = _sanitize_value(get_secret("KRAKEN_API_SECRET"))
    except SecretNotFound:
        sm_secret = ""

    if not key and sm_key:
        key = sm_key
        key_source = "secrets_manager"

    if not secret_raw and sm_secret:
        secret_raw = sm_secret
        secret_source = "secrets_manager"

    if not key:
        key_config = _sanitize_value(CONFIG.get("kraken_api_key") or "")
        if key_config:
            key = key_config
            key_source = "config"

    if not secret_raw:
        secret_config = _sanitize_value(CONFIG.get("kraken_api_secret") or "")
        if secret_config:
            secret_raw = secret_config
            secret_source = "config"

    if not key or not secret_raw:
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

    CONFIG["_kraken_key_origin"] = key_source or "unknown"
    CONFIG["_kraken_secret_origin"] = secret_source or "unknown"
    CONFIG["kraken_api_key"] = key
    CONFIG["kraken_api_secret"] = secret
    os.environ.setdefault("KRAKEN_API_KEY", key)
    os.environ.setdefault("KRAKEN_API_SECRET", secret)

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

    error_types = (KrakenAPIError, RuntimeError, ValueError)

    for pair in candidates:
        try:
            meta = _kraken_pair_meta(pair)
        except error_types as exc:  # pragma: no cover
            logger.warning(
                "Failed to validate pair %s via Kraken metadata: %s",
                pair,
                exc,
            )
            rejected.append(pair)
            continue

        if meta is None:
            logger.warning(
                "Kraken metadata unavailable; accepting pair %s without validation",
                pair,
            )
            validated.append(pair)
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

    logger.critical(
        "No tradable pairs validated; falling back to defaults without metadata confirmation.",
    )
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

    error_types = (KrakenAPIError, RuntimeError, ValueError)

    for pair in pairs:
        try:
            meta = _kraken_pair_meta(pair)
        except error_types as exc:  # pragma: no cover
            logger.warning(
                "Failed to load trade size metadata for %s: %s — using defaults",
                pair,
                exc,
            )
            continue
        if meta is None:
            logger.warning("Kraken metadata unavailable; using fallback trade sizes for %s", pair)
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


_DOTENV_VALUES = _load_dotenv_into_env()

_ensure_env_credentials()

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

if not CONFIG.get("kraken_api_key") or not CONFIG.get("kraken_api_secret"):
    sources_hint = []
    if _DOTENV_VALUES:
        sources_hint.append(".env")
    if os.getenv("KRAKEN_API_KEY") or os.getenv("KRAKEN_API_SECRET"):
        sources_hint.append("environment")
    logger.error(
        "Kraken API credentials are missing. Checked sources: %s",
        ", ".join(sources_hint) or "environment",
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

_confirmation_env = os.getenv("LIVE_CONFIRMATION_FILE")
if _confirmation_env:
    CONFIRMATION_CANDIDATE = _sanitize_value(_confirmation_env)
    if not CONFIRMATION_CANDIDATE:
        CONFIRMATION_CANDIDATE = _DEFAULT_LIVE_CONFIRMATION_FILE
else:
    CONFIRMATION_CANDIDATE = _DEFAULT_LIVE_CONFIRMATION_FILE

CONFIRMATION_PATH = str(Path(CONFIRMATION_CANDIDATE).expanduser())

live_mode_cfg = CONFIG.setdefault("live_mode", {})
live_mode_cfg["confirmation_file"] = CONFIRMATION_PATH

_LIVE_FORCE_ENV = _to_bool(os.getenv("LIVE_FORCE"), False)
if _LIVE_FORCE_ENV:
    logger.warning("LIVE_FORCE override enabled via environment variable.")

live_mode_cfg["force_override"] = _LIVE_FORCE_ENV

_risk_state_env = os.getenv("RISK_GUARD_STATE_FILE")
if _risk_state_env:
    RISK_CANDIDATE = _sanitize_value(_risk_state_env)
    if not RISK_CANDIDATE:
        RISK_CANDIDATE = "logs/risk_guard_state.json"
else:
    RISK_CANDIDATE = "logs/risk_guard_state.json"

RISK_STATE_PATH = str(Path(RISK_CANDIDATE).expanduser())
live_mode_cfg["risk_state_file"] = RISK_STATE_PATH

_risk_drawdown_env = os.getenv("LIVE_RISK_DRAWDOWN_THRESHOLD")
_risk_failure_env = os.getenv("LIVE_RISK_FAILURE_LIMIT")

drawdown_threshold_default = max(_to_float(_risk_drawdown_env, 0.10), 0.0)
failure_limit_default = max(_to_int(_risk_failure_env, 5), 1)

live_mode_cfg.setdefault("drawdown_threshold", drawdown_threshold_default)
live_mode_cfg.setdefault("failure_limit", failure_limit_default)

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
    "query_api_key_permissions",
]
