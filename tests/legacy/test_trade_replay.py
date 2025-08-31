"""Unit test for TradeReplay functionality."""

import os
import json
from datetime import datetime, timedelta, UTC
import pandas as pd
from crypto_trading_bot.replay.trade_replay import TradeReplay, DATA_DIR


def test_trade_replay_creates_png(tmp_path):
    """
    Unit test for TradeReplay.
    Ensures that given a fake trade and OHLCV data,
    a replay PNG is successfully generated.
    """

    # --- Setup fake ledger ---
    ledger_path = tmp_path / "trades.log"
    trade = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "pair": "FAKE/USD",
        "price": 100,
        "strategy": "TestStrategy",
    }
    with open(ledger_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(trade) + "\n")

    # --- Setup fake OHLCV data ---
    os.makedirs(DATA_DIR, exist_ok=True)
    fake_csv = os.path.join(DATA_DIR, "FAKE-USD.csv")
    now = datetime.now(UTC)
    df = pd.DataFrame({
        "timestamp": pd.date_range(now - timedelta(minutes=5), periods=10, freq="min"),
        "open": [100 + i for i in range(10)],
        "high": [101 + i for i in range(10)],
        "low": [99 + i for i in range(10)],
        "close": [100 + i for i in range(10)],
        "volume": [10] * 10,
    })
    df.to_csv(fake_csv, index=False)

    # --- Run TradeReplay ---
    replay = TradeReplay(ledger_path=str(ledger_path))
    replay.plot_trade(replay.trades[0])  # generates PNG

    # --- Assertions ---
    files = os.listdir(os.path.join("reports", "replays"))
    matching_files = [f for f in files if f.startswith("trade_FAKE-USD") and f.endswith(".png")]
    assert matching_files, "Replay PNG was not created!"
