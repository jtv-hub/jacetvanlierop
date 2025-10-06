"""Live trading confirmation gate utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from crypto_trading_bot import config as bot_config
from crypto_trading_bot.config import CONFIG, ConfigurationError

logger = logging.getLogger(__name__)

_DEFAULT_CONFIRMATION_FILENAME = ".confirm_live_trade"


def _resolve_confirmation_path() -> Path:
    """Return the path to the live confirmation sentinel file."""

    live_mode_cfg = CONFIG.get("live_mode", {}) or {}
    candidate = live_mode_cfg.get("confirmation_file") or os.getenv(
        "LIVE_CONFIRMATION_FILE",
        _DEFAULT_CONFIRMATION_FILENAME,
    )
    return Path(str(candidate)).expanduser().resolve()


def _force_override_enabled() -> bool:
    live_mode_cfg = CONFIG.get("live_mode", {}) or {}
    force_flag = bool(live_mode_cfg.get("force_override"))
    if force_flag and not live_mode_cfg.get("_force_logged"):
        logger.warning("LIVE_FORCE override active — skipping live confirmation gate (use only for CI/tests).")
        live_mode_cfg["_force_logged"] = True
    return force_flag


def require_live_confirmation(raise_on_fail: bool = True) -> bool:
    """Enforce confirmation sentinel before allowing live-mode actions.

    Returns ``True`` when live mode is either disabled or the confirmation
    file is present (or bypassed via ``LIVE_FORCE``). When the confirmation
    file is missing, logs a critical message and either raises a
    ``ConfigurationError`` or returns ``False`` depending on
    ``raise_on_fail``.
    """

    if not bot_config.is_live:
        return True

    if _force_override_enabled():
        return True

    confirmation_path = _resolve_confirmation_path()

    if confirmation_path.exists():
        if not bot_config.CONFIG.get("live_mode", {}).get("confirmation_acknowledged"):
            bot_config.CONFIG.setdefault("live_mode", {})["confirmation_acknowledged"] = True
            logger.info(
                "Live confirmation acknowledged at %s.",
                confirmation_path,
            )
        return True

    message = (
        "Live trading blocked: confirmation file missing. "
        f"Expected sentinel at {confirmation_path}. Run scripts/enable_live.sh to proceed."
    )
    logger.critical(message)

    if raise_on_fail:
        raise ConfigurationError(message)
    return False


__all__ = ["require_live_confirmation"]
