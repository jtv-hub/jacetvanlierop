"""
portfolio_state.py

Manages current reinvestment rate based on market conditions and portfolio value.
Refreshes the reinvestment rate once per day using logic from reinvestment.py.
"""

import json
import os
from datetime import datetime

from crypto_trading_bot.bot.utils.reinvestment import calculate_reinvestment_rate

STATE_FILE = "state/reinvestment_state.json"


# You can replace this with your actual portfolio tracker
def get_current_portfolio_value() -> float:
    """Mock function to retrieve current portfolio value. Replace with real implementation."""
    # Placeholder — replace with actual logic
    return 100_000.0


# You can replace this with your actual market regime detector
def get_current_market_regime() -> str:
    """Mock function to determine current market regime. Replace with real implementation."""
    # Placeholder — replace with actual logic
    return "trending"


def load_state():
    """Loads the current reinvestment state from a local JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}


def save_state(state):
    """Saves the reinvestment state to a local JSON file."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


def get_reinvestment_rate() -> float:
    """Calculates or retrieves the reinvestment rate for the current day."""
    state = load_state()
    today = datetime.now().strftime("%Y-%m-%d")

    # Refresh if stale or never computed
    if state.get("last_updated") != today:
        portfolio_value = get_current_portfolio_value()
        market_regime = get_current_market_regime()
        new_rate = calculate_reinvestment_rate(portfolio_value, market_regime)

        state["last_updated"] = today
        state["reinvestment_rate"] = new_rate
        save_state(state)

    return state.get("reinvestment_rate", 0.0)


def example_usage():
    """Prints the current reinvestment rate for debugging or standalone use."""
    rate = get_reinvestment_rate()
    print(f"📈 Current reinvestment rate: {rate:.0%}")


if __name__ == "__main__":
    example_usage()
