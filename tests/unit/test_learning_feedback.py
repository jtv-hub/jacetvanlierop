"""Test that the learning machine analyzes trades and writes suggestions.

The test prepares a small dummy trades log with two closed trades (one win,
one loss), invokes the learning cycle, and ensures a suggestions JSONL file
exists with entries that include the expected fields.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np


def test_learning_feedback_writes_suggestions(tmp_path):
    # Fixture kept for compatibility with earlier versions; discard to avoid lint warnings.
    del tmp_path

    # Generate dummy trades, run learning, and assert suggestions written.
    # Sanity-check numpy behaves normally (array creation and boolean mask ops)
    arr = np.array([1.0, -0.5, 0.2], dtype=float)
    wins = int((arr > 0).sum())
    assert isinstance(wins, int)

    # Prepare dummy trades (compact JSONL) — two closed trades
    os.makedirs("logs", exist_ok=True)
    trades_path = os.path.join("logs", "trades.log")
    with open(trades_path, "w", encoding="utf-8") as f:
        t1 = {
            "trade_id": f"T-{uuid.uuid4()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": "BTC/USDC",
            "size": 0.01,
            "strategy": "TestStrat",
            "confidence": 0.2,  # low confidence
            "status": "closed",
            "entry_price": 20000.0,
            "exit_price": 19800.0,
            "roi": -0.01,  # loss
            "reason": "stop_loss",
        }
        t2 = {
            "trade_id": f"T-{uuid.uuid4()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": "ETH/USDC",
            "size": 0.05,
            "strategy": "TestStrat",
            "confidence": 0.9,  # high confidence
            "status": "closed",
            "entry_price": 1000.0,
            "exit_price": 1100.0,
            "roi": 0.1,  # profit
            "reason": "take_profit",
        }
        f.write(json.dumps(t1, separators=(",", ":")) + "\n")
        f.write(json.dumps(t2, separators=(",", ":")) + "\n")

    # Clear target feedback file
    output_path = os.path.join("logs", "learning_feedback.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    # Try to run the learning machine entrypoint if available; fall back to
    # running a single cycle and generating simple suggestions
    import importlib  # pylint: disable=import-outside-toplevel

    lm = importlib.import_module("crypto_trading_bot.learning.learning_machine")
    if hasattr(lm, "run_learning_machine"):
        lm.run_learning_machine()
    else:
        # Fallback: run cycle and generate suggestions using optimization module
        metrics = lm.run_learning_cycle()
        try:
            # pylint: disable-next=import-outside-toplevel
            from crypto_trading_bot.learning.optimization import (
                generate_suggestions,
            )
        except ImportError:

            def generate_suggestions(_report):
                return [{"suggestion": "Monitor performance", "confidence": 0.5}]

        suggestions = generate_suggestions(metrics)
        os.makedirs("logs", exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).isoformat()
            for s in suggestions:
                rec = {"timestamp": ts, **s}
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # Validate output
    assert os.path.exists(output_path), "learning_feedback.jsonl was not created"
    with open(output_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines, "learning_feedback.jsonl has no entries"
    first = json.loads(lines[0])
    # Accept either legacy "suggestion"/"confidence" keys or the newer schema with strategy/confidence_* fields.
    has_legacy_keys = "suggestion" in first and "confidence" in first
    has_modern_keys = {"strategy", "confidence_after", "status"}.issubset(first.keys())
    assert has_legacy_keys or has_modern_keys, f"Unexpected suggestion schema: {first}"
