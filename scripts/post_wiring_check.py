"""Quick manual check for Kraken credential wiring.

- Enables live mode (validation only) to ensure credentials pass sanitation.
- Fetches balance via market_data helper (should raise on auth issues).
- Attempts validate-only order to confirm signing path returns structured response.
"""

from __future__ import annotations

import json
import logging

from crypto_trading_bot.bot.market_data import BalanceFetchError, get_account_balance
from crypto_trading_bot.config import CONFIG, ConfigurationError, set_live_mode
from crypto_trading_bot.utils.kraken_client import kraken_place_order

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("post_wiring_check")


def mask(value: str) -> str:
    if not value:
        return "***"
    return value[:6] + "***"


def main() -> int:
    key_origin = CONFIG.get("_kraken_key_origin", "unknown")
    secret_origin = CONFIG.get("_kraken_secret_origin", "unknown")
    logger.info(
        "Credential origins | key_origin=%s secret_origin=%s key_prefix=%s secret_length=%d",
        key_origin,
        secret_origin,
        mask(CONFIG.get("kraken_api_key", "")),
        len(CONFIG.get("kraken_api_secret", "")),
    )

    try:
        set_live_mode(True)
        logger.info("Live mode enabled for verification.")
    except ConfigurationError as exc:
        logger.error("Live mode failed: %s", exc)
        return 1

    try:
        balance = get_account_balance()
        logger.info("Balance fetch response: %s", balance)
    except BalanceFetchError as exc:
        logger.error("Balance fetch failed: %s", exc)

    kraken_cfg = CONFIG.get("kraken", {}) or {}
    min_cost_default = float(kraken_cfg.get("min_cost_threshold", 0.5))
    pair_thresholds = kraken_cfg.get("pair_cost_minimums", {}) or {}
    usd_pair_threshold = float(pair_thresholds.get("USDC/USD", min_cost_default))

    order_response = kraken_place_order(
        "USDC/USD",
        "buy",
        1,
        price=0.99,
        validate=True,
        min_cost_threshold=usd_pair_threshold,
    )
    logger.info("Order response:\n%s", json.dumps(order_response, indent=2, default=str))

    tiny_order_response = kraken_place_order(
        "USDC/USD",
        "buy",
        0.0001,
        price=0.01,
        validate=True,
        min_cost_threshold=usd_pair_threshold,
    )
    logger.info(
        "Tiny order response (expected cost-minimum):\n%s",
        json.dumps(tiny_order_response, indent=2, default=str),
    )

    set_live_mode(False)
    logger.info("Post-wiring check complete; live mode disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
