"""
Integration test for the nightly pipeline.
Creates mock data, runs the full pipeline, and checks outputs.
"""

import json
import os

from scripts import nightly_pipeline

# === Setup directories ===
os.makedirs("ledger", exist_ok=True)
os.makedirs("learning", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Step 1: Seed a mock trade ledger ===
LEDGER_FILE = "ledger/trade_ledger.json"
mock_trades = [
    {
        "timestamp": "2025-08-16 12:00:00",
        "pair": "BTC/USDC",
        "signal": "BUY",
        "confidence": 0.8,
        "regime": "trend",
        "outcome": "win",
    },
    {
        "timestamp": "2025-08-16 12:05:00",
        "pair": "ETH/USDC",
        "signal": "SELL",
        "confidence": 0.6,
        "regime": "chop",
        "outcome": "loss",
    },
]

with open(LEDGER_FILE, "w", encoding="utf-8") as f:
    json.dump(mock_trades, f)

print("✅ Mock ledger created.")

# === Step 2: Seed a mock shadow test results file ===
RESULTS_FILE = "learning/shadow_test_results.jsonl"
mock_results = [
    {
        "strategy_name": "RSI",
        "param_change": {"rsi_upper": 65},
        "confidence": 0.75,
        "status": "pass",
    },
    {
        "strategy_name": "MACD",
        "param_change": {"fast": 10, "slow": 30},
        "confidence": 0.55,
        "status": "fail",
    },
]

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    for r in mock_results:
        f.write(json.dumps(r) + "\n")

print("✅ Mock shadow test results created.")

# === Step 3: Run the full pipeline ===
print("\n🚀 Running nightly pipeline...")
nightly_pipeline.run_pipeline()

# === Step 4: Verify outputs ===
DECISIONS_FILE = "learning/final_decisions.jsonl"
if os.path.exists(DECISIONS_FILE):
    with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print("\n✅ Gatekeeper decisions recorded:")
        for line in lines:
            print("  ", line.strip())
else:
    print("\n❌ No final decisions found!")
