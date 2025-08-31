#!/usr/bin/env python3
"""
Ingest paper trades into the main trade ledger.

- Reads all JSONL trade files from logs/paper/
- Validates trade structure
- Appends them into ledger/trade_ledger.json using log_trade()
- Marks source = "paper" so they are distinguishable from live trades
"""

import json
import sys
from pathlib import Path

# Use absolute import for project safety
try:
    from crypto_trading_bot.ledger.trade_ledger import log_trade
except ImportError as e:
    print(f"[ingest] ERROR: failed to import trade_ledger: {e}", file=sys.stderr)
    sys.exit(1)

# Use project-root safe path
BASE_DIR = Path(__file__).resolve().parents[1]
PAPER_LOG_DIR = BASE_DIR / "logs" / "paper"


def ingest_paper_trades() -> int:
    """
    Read paper trades from logs/paper and ingest them into the trade ledger.
    """
    if not PAPER_LOG_DIR.exists():
        print(f"[ingest] No paper trades directory found at {PAPER_LOG_DIR}")
        return 0

    trade_files = sorted(PAPER_LOG_DIR.glob("paper_trades_*.jsonl"))
    if not trade_files:
        print("[ingest] No paper trades found to ingest.")
        return 0

    total = 0
    for fpath in trade_files:
        with fpath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    trade = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines

                # Validate expected fields
                required = {"symbol", "entry", "exit", "pnl_pct", "exit_reason"}
                if not required.issubset(trade):
                    continue

                # Call ledger log_trade with extended fields
                try:
                    log_trade(
                        trading_pair=trade.get("symbol"),
                        trade_size=trade.get("entry"),
                        strategy_name="paper_trade",
                        confidence_score=trade.get("pnl_pct", 0),
                        extra_fields={
                            "exit": trade.get("exit"),
                            "pnl_pct": trade.get("pnl_pct"),
                            "exit_reason": trade.get("exit_reason"),
                            "ts_start": trade.get("ts_start"),
                            "ts_end": trade.get("ts_end"),
                            "source": "paper",
                            "file_source": trade.get("file_source"),
                        },
                    )
                    total += 1
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ) as e:
                    print(
                        f"[ingest] Failed to log trade: {e}",
                        file=sys.stderr,
                    )
                    continue

    print(f"[ingest] Ingested {total} paper trades into ledger.")
    return total


def run() -> None:
    """Entry point wrapper for pipeline usage."""
    ingest_paper_trades()


if __name__ == "__main__":
    sys.exit(0 if ingest_paper_trades() else 1)
