"""
One-off script to insert dummy trades into logs/trades.log
for testing the learning machine and scheduler.
"""

import json
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Dummy trades with ROI
dummy_trades = [
    {
        "timestamp": "2025-08-19 01:00:00",
        "pair": "BTC-USD",
        "size": 100,
        "strategy": "SimpleRSI",
        "confidence": 0.85,
        "status": "executed",
        "roi": 0.05,
    },
    {
        "timestamp": "2025-08-19 01:05:00",
        "pair": "ETH-USD",
        "size": 50,
        "strategy": "DualThreshold",
        "confidence": 0.75,
        "status": "executed",
        "roi": -0.02,
    },
    {
        "timestamp": "2025-08-19 01:10:00",
        "pair": "BTC-USD",
        "size": 120,
        "strategy": "SimpleRSI",
        "confidence": 0.90,
        "status": "executed",
        "roi": 0.03,
    },
]

# Append to trades.log
with open("logs/trades.log", "a", encoding="utf-8") as f:
    for trade in dummy_trades:
        f.write(json.dumps(trade) + "\n")

print("✅ Dummy trades inserted into logs/trades.log")
