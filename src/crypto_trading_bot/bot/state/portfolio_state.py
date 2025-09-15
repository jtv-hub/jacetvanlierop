"""Persistent portfolio state helpers used for daily compounding.

This module keeps track of available trading capital based on realized
trades, derives a reinvestment rate, and stores the latest portfolio
snapshot so other services (scheduler, trading logic, reporting) can
pull a consistent view of capital.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from crypto_trading_bot.analytics.roi_calculator import compute_running_balance
from crypto_trading_bot.bot.utils.reinvestment import calculate_reinvestment_rate

STATE_FILE = "data/portfolio_state.json"
TRADES_LOG_PATH = "logs/trades.log"
DEFAULT_STARTING_BALANCE = 100_000.0


def _parse_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp into a timezone-aware datetime."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def load_closed_trades(log_path: str = TRADES_LOG_PATH) -> List[dict]:
    """Return closed trades with valid ROI information from the trade log."""
    if not os.path.exists(log_path):
        return []

    closed: List[dict] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                trade = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = (trade.get("status") or "").lower()
            roi = trade.get("roi")
            size = trade.get("size")
            if status == "closed" and isinstance(roi, (int, float)) and isinstance(size, (int, float)):
                closed.append(trade)
    return closed


def load_state() -> Dict[str, Any]:
    """Read the persisted portfolio state from disk."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    """Persist the portfolio state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=4)


def get_current_market_regime() -> str:
    """Placeholder hook for regime detection integration."""
    # TODO: replace with real regime detection once available
    return "trending"


def refresh_portfolio_state(
    *,
    log_path: str = TRADES_LOG_PATH,
    starting_balance: float = DEFAULT_STARTING_BALANCE,
) -> Dict[str, Any]:
    """Recompute and persist the latest portfolio snapshot."""
    closed_trades = load_closed_trades(log_path)
    available_capital = compute_running_balance(closed_trades, starting_balance)

    most_recent_close: datetime | None = None
    for trade in closed_trades:
        ts = _parse_timestamp(trade.get("timestamp"))
        if ts is None:
            continue
        if most_recent_close is None or ts > most_recent_close:
            most_recent_close = ts

    regime = get_current_market_regime()
    reinvestment_rate = calculate_reinvestment_rate(available_capital, regime)

    snapshot: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available_capital": round(float(available_capital), 2),
        "reinvestment_rate": float(reinvestment_rate),
        "last_close_date": most_recent_close.date().isoformat() if most_recent_close else None,
        "market_regime": regime,
        "starting_balance": float(starting_balance),
        "closed_trade_count": len(closed_trades),
    }
    save_state(snapshot)
    return snapshot


def load_portfolio_state(
    *,
    refresh: bool = False,
    log_path: str = TRADES_LOG_PATH,
    starting_balance: float = DEFAULT_STARTING_BALANCE,
) -> Dict[str, Any]:
    """Load the portfolio state, recomputing it if requested or missing."""
    if refresh:
        return refresh_portfolio_state(log_path=log_path, starting_balance=starting_balance)

    state = load_state()
    if "available_capital" not in state:
        return refresh_portfolio_state(log_path=log_path, starting_balance=starting_balance)
    return state


def get_reinvestment_rate(
    *,
    refresh: bool = False,
    log_path: str = TRADES_LOG_PATH,
    starting_balance: float = DEFAULT_STARTING_BALANCE,
) -> float:
    """Convenience accessor for the current reinvestment rate."""
    state = load_portfolio_state(
        refresh=refresh,
        log_path=log_path,
        starting_balance=starting_balance,
    )
    return float(state.get("reinvestment_rate", 0.0))


def example_usage() -> None:
    """Print the latest snapshot for quick manual inspection."""
    state = load_portfolio_state(refresh=True)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    example_usage()
