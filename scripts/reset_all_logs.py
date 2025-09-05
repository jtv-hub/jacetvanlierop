"""
Reset All Logs Script

Safely resets trades, positions, portfolio state, and optionally seeded
price history used for RSI/trend warmup.

Usage:
  python scripts/reset_all_logs.py --wipe-history
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob


def wipe_file(path: str) -> bool:
    """Clear a file if it exists (truncate to zero length). Returns True if touched."""
    try:
        if os.path.exists(path):
            with open(path, "w", encoding="utf-8"):
                pass
            print(f"✅ Cleared {path}")
            return True
        else:
            print(f"ℹ️  {path} not found (already clean)")
            return False
    except OSError as e:
        print(f"⚠️  Failed to clear {path}: {e}")
        return False


def wipe_jsonl(path: str) -> None:
    """Remove all lines from a JSONL file (truncate)."""
    wipe_file(path)


def reset_portfolio_state(path: str = "logs/portfolio_state.json") -> None:
    """Reset portfolio state to sensible defaults."""
    state = {
        "daily_trade_count": 0,
        "win_streak_count": 0,
        "last_trade_day": None,
        "capital_buffer": 0.25,
        "regime": "unknown",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print("✅ Reset portfolio_state.json")


def reset_logs(wipe_history: bool = False) -> None:
    """Reset trades, positions, rotating logs and optionally seeded history."""
    os.makedirs("logs", exist_ok=True)

    # Wipe key files (truncate or create) with explicit targets
    key_files = [
        # Core trade/positions/state
        "logs/trades.log",
        "src/logs/trades.log",  # legacy location if present
        "logs/positions.jsonl",
        "positions.jsonl",  # safety for alternate path
        "logs/portfolio_state.json",
        # System logs
        "logs/anomalies.log",
        "logs/system.log",
        "logs/alerts.log",
        "logs/exit_check.log",
        # Learning logs
        "logs/learning_feedback.jsonl",
        "logs/shadow_test_results.jsonl",
    ]
    for p in key_files:
        # JSONL are just files; truncation is safe for both .log and .jsonl
        wipe_file(p)

    reset_portfolio_state()

    # Clear .log files in logs/ except those explicitly preserved
    for log_file in glob("logs/*.log"):
        # Preserve audit-style logs if desired
        if "audit" in os.path.basename(log_file):
            continue
        wipe_file(log_file)

    if wipe_history:
        # Wipe seeded price history used for startup warmup and caches
        history_targets = [
            "data/seeded_prices.json",
            "data/price_cache.json",
            "price_cache.json",
        ]
        for hist_file in history_targets:
            wipe_file(hist_file)
        print("⚠️ Price history wiped — RSI may fail until reseeded.")

    print("✅ All logs, positions, and price history wiped successfully. " "Ready for clean trading.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset bot logs and state.")
    parser.add_argument(
        "--wipe-history",
        action="store_true",
        help="Also wipe seeded price history (reseed before running)",
    )
    args = parser.parse_args()
    reset_logs(wipe_history=args.wipe_history)


if __name__ == "__main__":
    main()
