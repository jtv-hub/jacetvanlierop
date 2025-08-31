"""
Daily Heartbeat Script

Runs scheduled daily maintenance tasks for the crypto trading bot,
including reinvestment rate refresh
and other periodic health checks or cleanup routines.
"""

import sys
import traceback

from crypto_trading_bot.bot.state.portfolio_state import (
    get_reinvestment_rate,
)


def run_daily_tasks():
    """Run all daily scheduled maintenance tasks for the trading bot."""
    print("🔁 Running daily heartbeat tasks...")

    try:
        # Refresh reinvestment rate (automatically refreshes if outdated)
        rate = get_reinvestment_rate()
        print(f"📊 Reinvestment rate refreshed: {rate:.0%}")

        # Placeholder: Add other tasks like log cleanup, anomaly review, etc.
        print("✅ Daily tasks completed successfully.")

    except Exception:  # pylint: disable=broad-except
        print("❌ Error during daily heartbeat:")
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    run_daily_tasks()
