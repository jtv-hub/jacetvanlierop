# trade_replay.py
"""
Trade Replay Module
Reconstructs trades visually using OHLCV data and ledger entries.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "data"
REPORTS_DIR = "reports/replays"
os.makedirs(REPORTS_DIR, exist_ok=True)


class TradeReplay:
    """Handles loading trades and generating replay visualizations from OHLCV data."""

    def __init__(self, ledger_path: str = "logs/trades.log"):
        self.ledger_path = ledger_path
        self.trades = self.load_trades()

    def load_trades(self):
        """Load all trades from the ledger file."""
        trades = []
        if not os.path.exists(self.ledger_path):
            print(f"❌ Ledger not found at {self.ledger_path}")
            return trades

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(trades)} trades from ledger.")
        return trades

    def load_candles(self, pair: str, start: str, end: str):
        """Load OHLCV data for the trading pair."""
        safe_pair = pair.replace("/", "-")  # sanitize for filename
        filename = f"{safe_pair}.csv"
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"No OHLCV data found at {path}")

        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        return df

    def plot_trade(self, trade: dict):
        """Plot a single trade with OHLCV data."""
        # Handle both old and new formats for "pair"
        pair = trade.get("pair")
        if isinstance(pair, dict):
            pair = pair.get("pair", "UNKNOWN")

        start_time = trade["timestamp"]
        end_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        df = self.load_candles(pair, start_time, end_time)

        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["close"], label="Close Price")
        plt.axvline(pd.to_datetime(start_time), color="g", linestyle="--", label="Entry")

        # Optional: mark price point
        if "price" in trade:
            entry_price = trade["price"]
            plt.axhline(entry_price, color="r", linestyle=":", label=f"Entry @ {entry_price}")

        strategy_name = trade.get("strategy", "UnknownStrategy")

        plt.title(f"Trade Replay - {pair} ({strategy_name})")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        # Save replay
        ts = start_time.replace(":", "").replace(" ", "_")
        safe_pair = pair.replace("/", "-")
        out_path = os.path.join(REPORTS_DIR, f"trade_{safe_pair}_{ts}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Replay saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Replay CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="Replay the most recent trade")
    group.add_argument("--all", action="store_true", help="Replay all trades")
    group.add_argument("--trade", type=int, help="Replay a specific trade by index")

    args = parser.parse_args()
    replay = TradeReplay()

    if not replay.trades:
        print("❌ No trades to replay.")
    elif args.latest:
        replay.plot_trade(replay.trades[-1])
    elif args.all:
        for t in replay.trades:
            try:
                replay.plot_trade(t)
            except FileNotFoundError as e:
                print(f"⚠️ Skipping trade: {e}")
    elif args.trade is not None:
        index = args.trade
        if 0 <= index < len(replay.trades):
            replay.plot_trade(replay.trades[index])
        else:
            print(f"❌ Invalid trade index: {index}")
