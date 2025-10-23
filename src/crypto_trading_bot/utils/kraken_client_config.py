"""Configuration loading helpers for Kraken client utilities."""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType
from typing import Any, Dict

from crypto_trading_bot.utils.helpers import sanitize_base64_secret as _helpers_sanitize_base64_secret

logger = logging.getLogger(__name__)


class ConfigState:
    """Track cached configuration module references and helpers."""

    def __init__(self) -> None:
        self.module: ModuleType | None = None
        self.import_failed: bool = False
        self.config: Dict[str, Any] = {}
        self.sanitize = _helpers_sanitize_base64_secret


CONFIG_STATE = ConfigState()
CONFIG = CONFIG_STATE.config


def sanitize_base64_secret(secret: str, *, strict: bool = False) -> str:
    """Return a sanitized Kraken secret using the active config helper."""

    return CONFIG_STATE.sanitize(secret, strict=strict)


def ensure_config_loaded() -> None:
    """Import ``crypto_trading_bot.config`` lazily and cache references."""

    if CONFIG_STATE.module is not None or CONFIG_STATE.import_failed:
        return

    module = sys.modules.get("crypto_trading_bot.config")
    if module is None:
        try:
            module = importlib.import_module("crypto_trading_bot.config")
        except ImportError as exc:  # pragma: no cover - defensive logging
            CONFIG_STATE.import_failed = True
            logger.debug("Config module unavailable: %s", exc)
            return

    if not isinstance(module, ModuleType):  # pragma: no cover - defensive
        CONFIG_STATE.import_failed = True
        logger.debug("Config module reference is not a module: %r", module)
        return

    CONFIG_STATE.module = module
    config_mapping = getattr(module, "CONFIG", None)
    if isinstance(config_mapping, dict):
        CONFIG_STATE.config.clear()
        CONFIG_STATE.config.update(config_mapping)

    sanitize_fn = getattr(module, "_sanitize_base64_secret", None)
    if callable(sanitize_fn):
        CONFIG_STATE.sanitize = sanitize_fn  # type: ignore[assignment]


def get_config_module() -> ModuleType | None:
    """Return the cached configuration module, loading it if necessary."""

    ensure_config_loaded()
    return CONFIG_STATE.module
