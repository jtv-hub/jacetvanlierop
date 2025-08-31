"""
rejected_simulator.py

Simulates the performance of rejected strategy suggestions using historical
price data to evaluate how they would have performed if accepted.
"""

from datetime import datetime
from crypto_trading_bot.learning.shadow_test_logger import log_shadow_test_result


def simulate_rejected_strategy(
    suggestion_id, strategy_class, parameters, price_data, asset="BTC", timeframe="5m"
):
    """
    Simulates a rejected strategy on historical price data and logs the result.

    Args:
        suggestion_id (str): Unique ID of the rejected suggestion
        strategy_class (class): Strategy class to instantiate
        parameters (dict): Strategy parameters
        price_data (list): List of historical price points (floats)
        asset (str): Asset symbol (e.g., "BTC")
        timeframe (str): Timeframe label (e.g., "5m")
    """
    strategy = strategy_class(**parameters)
    trades = []
    entry_time = datetime.utcnow()

    for i in range(len(price_data)):
        window = price_data[: i + 1]
        if len(window) < max(parameters.values(), default=0):
            continue

        signal = strategy.generate_signal(window)
        if signal.get("signal") in ["buy", "sell"]:
            trades.append((i, signal["signal"]))

    exit_time = datetime.utcnow()
    trade_count = len(trades)
    wins = sum(1 for i, sig in trades if sig == "buy")  # Simulated "wins"
    win_rate = wins / trade_count if trade_count > 0 else 0
    roi = win_rate  # Placeholder for return-on-investment logic

    test_duration = int((exit_time - entry_time).total_seconds() / 60)

    log_shadow_test_result(
        suggestion_id,
        strategy_class.__name__,
        parameters,
        asset,
        timeframe,
        roi,
        win_rate,
        trade_count,
        test_duration,
    )

    return {
        "roi": roi,
        "win_rate": win_rate,
        "trades": trade_count,
        "duration": test_duration,
    }
