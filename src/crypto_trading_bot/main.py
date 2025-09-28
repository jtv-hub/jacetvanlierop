"""
Main entry point for the Crypto Trading Bot.
Allows user to run a one-off trade or start the scheduler.
"""

import argparse
import logging
import os

import crypto_trading_bot.config as bot_config
from crypto_trading_bot.bot.scheduler import run_scheduler
from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.config import CONFIG, ConfigurationError, get_mode_label, set_live_mode
from crypto_trading_bot.safety.prelaunch_guard import run_prelaunch_guard
from crypto_trading_bot.utils.system_checks import ensure_system_capacity

# Configure root logger for DEBUG output (ensures [RSI DEBUG] logs are visible)
logging.basicConfig(level=logging.DEBUG)

_GUARD_LOG_PATHS = ("logs/alerts.log", "logs/system.log")


def _ensure_logs_writable() -> None:
    """Fail fast if core logs cannot be opened, preventing silent live-mode runs without logging."""
    for path in _GUARD_LOG_PATHS:
        directory = os.path.dirname(path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("")
        except OSError as exc:
            logging.critical("Cannot open %s for append: %s", path, exc)
            raise SystemExit(1) from exc


def main():
    """
    Parses command-line arguments and runs the Crypto Trading Bot in the specified mode.
    Supports 'once' mode for a single trade
    and 'schedule' mode for continuous trading at set intervals.
    """
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["once", "schedule"],
        default="once",
        help=("Run mode: 'once' for single trade, 'schedule' for continuous trading"),
    )
    parser.add_argument("--size", type=float, default=100, help="Trade size in USD")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval in seconds for scheduler",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force validate-only mode (live trades send validate=True)",
    )
    parser.add_argument(
        "--confirm-live-mode",
        action="store_true",
        help=(
            "Explicit confirmation required to enable real trading. "
            "Without this flag the bot remains in paper mode regardless of environment."
        ),
    )

    args = parser.parse_args()

    try:
        ensure_system_capacity()
    except ConfigurationError as exc:
        logging.critical("System capacity check failed: %s", exc)
        raise SystemExit(1) from exc

    _ensure_logs_writable()

    if args.confirm_live_mode and args.dry_run:
        parser.error("--confirm-live-mode is incompatible with --dry-run")

    env_live_mode = os.getenv("LIVE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
    if env_live_mode:
        logging.warning("LIVE_MODE requested via environment variable.")

    if args.confirm_live_mode or env_live_mode:
        try:
            set_live_mode(True)
        except ConfigurationError as exc:
            logging.critical("Failed to enable live mode: %s", exc)
            raise SystemExit(1) from exc
    else:
        # Ensure we never inherit a lingering live flag from previous runs.
        set_live_mode(False)

    if args.dry_run:
        CONFIG.setdefault("live_mode", {})["dry_run"] = True
        CONFIG.setdefault("kraken", {})["validate_orders"] = True
        logging.warning("[DRY-RUN] Validate-only mode enabled via CLI flag — live orders will not fill.")
    elif CONFIG.get("live_mode", {}).get("dry_run"):
        logging.warning("[DRY-RUN] Validate-only mode enabled via configuration or environment.")

    current_live_flag = bot_config.is_live
    logging.info("Current trading mode: %s (is_live=%s)", get_mode_label(), current_live_flag)
    if not current_live_flag:
        logging.info("Paper mode active — live orders will be blocked.")
    else:
        logging.warning("🚨 Live trading enabled — orders will execute against real funds.")

    live_real_mode = current_live_flag and not bool(CONFIG.get("live_mode", {}).get("dry_run"))
    if live_real_mode:
        try:
            run_prelaunch_guard()
        except ConfigurationError as exc:
            logging.critical("Prelaunch guard failed: %s", exc)
            raise SystemExit(2) from exc

    if args.mode == "once":
        print("⚡ Running one-off evaluation...")
        evaluate_signals_and_trade()
        print("✅ Trade evaluation complete.")
    elif args.mode == "schedule":
        run_scheduler()


if __name__ == "__main__":
    main()
