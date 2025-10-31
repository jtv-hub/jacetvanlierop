"""Trading loop script.

Runs the main evaluation loop using the core trading logic module. It relies on
centralized configuration for tradable pairs and live pricing utilities defined
within the package. This script intentionally keeps orchestration minimal.
"""

import argparse
import logging
import os

from crypto_trading_bot.config import CONFIG, ConfigurationError, get_mode_label, is_live
from crypto_trading_bot.context.trading_context import TradingContext
from crypto_trading_bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.utils.system_checks import ensure_system_capacity

# Optional debug toggle configured via CONFIG["logging"]["debug_mode"] or env override.
_DEBUG_SENTINELS = {"1", "true", "yes", "on"}


def _coerce_debug(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _DEBUG_SENTINELS
    return bool(value)


_configured_debug = _coerce_debug(CONFIG.get("logging", {}).get("debug_mode", False))
_env_debug = os.getenv("CRYPTO_TRADING_BOT_DEBUG")
if _env_debug is not None:
    DEBUG_MODE = _coerce_debug(_env_debug)
else:
    DEBUG_MODE = _configured_debug

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)


def main(argv: list[str] | None = None):
    """
    Main function to run the trading loop with live data.
    """
    parser = argparse.ArgumentParser(description="Trading loop runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force validate-only mode (live trades become validate=True)",
    )
    args = parser.parse_args(argv)

    try:
        ensure_system_capacity()
    except ConfigurationError as exc:
        logging.critical("System capacity check failed: %s", exc)
        raise SystemExit(1) from exc

    if args.dry_run:
        CONFIG.setdefault("live_mode", {})["dry_run"] = True
        CONFIG.setdefault("kraken", {})["validate_orders"] = True
        logging.warning("[DRY-RUN] Validate-only mode enabled via --dry-run; no real fills will occur.")
    elif CONFIG.get("live_mode", {}).get("dry_run"):
        logging.warning("[DRY-RUN] Validate-only mode enabled via configuration or environment.")

    mode_label = get_mode_label()
    logging.info("🟢 Starting trading loop... (%s)", mode_label)
    if not is_live:
        logging.info("Running in paper mode — live trades are disabled by default.")
    else:
        logging.warning("🚨 LIVE MODE ENABLED — Orders will be sent to the exchange.")
    # Touch context so it is initialized for modules that rely on it,
    # but the trading logic manages and updates context internally.
    _ = TradingContext()
    evaluate_signals_and_trade()
    logging.info("✅ Trade evaluation completed.")


if __name__ == "__main__":
    main()
