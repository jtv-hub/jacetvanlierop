# src/crypto_trading_bot/config/__init__.py

CONFIG: dict = {
    # Centralized list of tradable pairs used across the app.
    # Added to ensure no hardcoded pairs scattered in the codebase.
    # Ordering matches requested scan order.
    "tradable_pairs": [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "LINK/USD",
    ],
    "rsi": {
        "period": 14,
        "lower": 48,
        "upper": 75,
        # Lower exit threshold for easier testing; tune in paper/live
        "exit_upper": 55,
    },
    "max_portfolio_risk": 0.10,
    "min_volume": 100,
    "trade_size": {"min": 0.001, "max": 0.005},
    "slippage": {
        "majors": 0.001,  # 0.1%
        "alts_min": 0.005,  # 0.5%
        "alts_max": 0.01,  # 1.0%
        "use_random": False,
    },
    "buffer_defaults": {
        "trending": 1.0,
        "chop": 0.5,
        "volatile": 0.5,
        "flat": 0.25,
        "unknown": 0.25,
    },
    "correlation": {"window": 30, "threshold": 0.7},
}
