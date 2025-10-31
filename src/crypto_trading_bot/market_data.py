"""
market_data.py

Production market data helpers. No mock fallbacks.

This module intentionally avoids importing any mock generators. If a live
price cannot be fetched, a warning is logged and None is returned.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from importlib import import_module
from typing import Callable, Optional

from crypto_trading_bot.config import CONFIG, is_live
from crypto_trading_bot.utils.kraken_client import kraken_get_balance
from crypto_trading_bot.utils.price_feed import get_current_price
from crypto_trading_bot.utils.system_logger import get_system_logger

system_logger = get_system_logger()
logger = system_logger.getChild("market_data")


class BalanceFetchError(RuntimeError):
    """Raised when no reliable balance source is available."""


def get_market_snapshot(trading_pair: str) -> Optional[dict]:
    """
    Fetch a minimal real-time market snapshot for a given trading pair.

    Returns a dict with timestamp, pair, and price on success; otherwise None.
    """
    price = get_current_price(trading_pair)
    if price is None:
        system_logger.warning("Live data unavailable for %s; returning None", trading_pair)
        return None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": trading_pair,
        "price": float(price),
    }


def _load_balance_provider() -> Optional[Callable[[], float]]:
    """Return a configured balance provider callable, if any."""

    provider_path = CONFIG.get("live_mode", {}).get("balance_provider")
    if not provider_path:
        return None
    target = provider_path.strip()
    if not target:
        return None
    if ":" not in target:
        logger.warning(
            "Balance provider '%s' is invalid; expected 'module:function' format.",
            target,
        )
        return None
    module_path, func_name = target.rsplit(":", 1)
    try:
        module = import_module(module_path)
        provider = getattr(module, func_name)
    except (ImportError, AttributeError) as exc:
        logger.warning("Unable to import balance provider %s: %s", target, exc)
        return None
    if not callable(provider):
        logger.warning("Configured balance provider '%s' is not callable.", target)
        return None
    return provider


def get_account_balance(*, use_mock_for_paper: bool = True) -> Optional[float]:
    """Return the latest account balance for live or paper mode.

    - When running in live mode, attempts the configured provider first,
      then falls back to an environment variable or configured constant.
    - In paper mode (default), returns the configured starting balance unless
      ``use_mock_for_paper`` is False.
    """

    paper_balance = float(CONFIG.get("paper_mode", {}).get("starting_balance", 100_000.0))

    if not is_live:
        if not use_mock_for_paper:
            logger.warning(
                "Balance branch=paper-mode override use_mock=False value=%.2f",
                paper_balance,
            )
        else:
            logger.info(
                "Balance branch=paper-mode source=config value=%.2f",
                paper_balance,
            )
        return paper_balance

    provider = _load_balance_provider()
    if provider is not None:
        try:
            balance = provider()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Balance provider call failed: %s", exc)
        else:
            try:
                value = float(balance)
            except (TypeError, ValueError):
                logger.warning("Provider returned non-numeric balance: %s", balance)
            else:
                if value >= 0:
                    provider_name = getattr(provider, "__name__", provider.__class__.__name__)
                    logger.info(
                        "Balance branch=provider source=%s value=%.2f",
                        provider_name,
                        value,
                    )
                    return value

    kraken_asset = (CONFIG.get("kraken", {}) or {}).get("balance_asset", "USDC")
    response = kraken_get_balance(kraken_asset)
    if response.get("ok"):
        value = float(response.get("balance", 0.0))
        logger.info(
            "Balance branch=kraken-live asset=%s value=%.8f",
            kraken_asset,
            value,
        )
        return value

    error_detail = response.get("error") or "unknown kraken balance error"
    if response.get("code") == "auth":
        message = f"Kraken authentication error: {error_detail}"
        logger.error("Balance branch=kraken-live auth-error=%s", message)
        raise BalanceFetchError(message)

    allow_fallback = bool(CONFIG.get("live_mode", {}).get("allow_balance_fallback", False))
    if not allow_fallback:
        logger.error(
            "Balance branch=kraken-live error=%s raw=%s fallback=disabled",
            error_detail,
            response.get("raw"),
        )
        raise BalanceFetchError("Live balance unavailable and fallback disabled.")

    logger.warning(
        "Balance branch=kraken-live error=%s raw=%s using fallback",
        error_detail,
        response.get("raw"),
    )

    env_var = CONFIG.get("live_mode", {}).get("balance_env_var") or "CRYPTO_TRADING_BOT_LIVE_BALANCE"
    env_value = os.getenv(env_var)
    if env_value:
        try:
            parsed = float(env_value)
            if parsed >= 0:
                logger.warning(
                    "Balance branch=env source=%s value=%.2f (fallback)",
                    env_var,
                    parsed,
                )
                return parsed
        except ValueError:
            logger.warning("Environment balance value %s is not numeric.", env_value)

    fallback = CONFIG.get("live_mode", {}).get("fallback_balance")
    try:
        fallback_val = float(fallback)
    except (TypeError, ValueError):
        fallback_val = None

    if fallback_val is not None and fallback_val > 0:
        logger.warning(
            "Balance branch=config-fallback value=%.2f",
            fallback_val,
        )
        return fallback_val

    message = "Live balance unavailable after fallback attempts."
    logger.error("Balance branch=error error=%s", message)
    raise BalanceFetchError(message)
