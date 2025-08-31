"""
Utility script to inspect the trade ledger.
Shows the most recent trades from logs/trades.log
"""

import json
import os

LOG_FILE = "logs/trades.log"


def check_ledger(n_last: int = 10):
    """
    Print the last n_last trades from the ledger for quick inspection.
    """
    if not os.path.exists(LOG_FILE):
        print(f"❌ Ledger file {LOG_FILE} not found.")
        return

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("ℹ️ Ledger is empty — no trades logged yet.")
        return

    print(f"\n📑 Showing last {min(n_last, len(lines))} trades in ledger:\n")
    for line in lines[-n_last:]:
        try:
            trade = json.loads(line.strip())
            print(
                f"[{trade['timestamp']}] {trade['pair']} | "
                f"Strategy={trade.get('strategy_name', trade.get('strategy', 'N/A'))} | "
                f"Signal={trade.get('signal', 'N/A')} | "
                f"PnL={trade.get('pnl', 0):.2f} | ROI={trade.get('roi', 0):.4f} | "
                f"Regime={trade.get('regime', 'N/A')} | Status={trade.get('status', 'N/A')}"
            )
        except json.JSONDecodeError:
            print("⚠️ Skipped corrupt line:", line.strip())


if __name__ == "__main__":
    check_ledger(n_last=10)
