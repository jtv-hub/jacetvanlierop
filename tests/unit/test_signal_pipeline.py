"""Unit tests for the lightweight signal gathering helper.

These tests validate shape and basic properties of the output from
`gather_signals`, not the full end-to-end trading pipeline.
"""

from datetime import datetime  # standard library first

from crypto_trading_bot.bot.trading_logic import gather_signals


def test_gather_signals_rsi_valid():
    """gather_signals returns dict with rsi and raw fields for valid inputs."""
    # Simulate an uptrend sequence and adequate volumes
    prices = [100 + i for i in range(30)]
    volumes = [1500] * 30

    ctx = {"timestamp": datetime.utcnow()}  # optional context

    out = gather_signals(prices, volumes, ctx=ctx)

    # Shape checks
    assert isinstance(out, dict)
    assert "rsi" in out and "raw" in out
    assert isinstance(out["raw"], dict)
    assert out["raw"].get("prices") is prices
    assert out["raw"].get("volumes") is volumes

    # RSI should be a float within [0, 100] or None if calc failed
    rsi = out.get("rsi")
    assert rsi is None or (isinstance(rsi, (int, float)) and 0 <= rsi <= 100)
