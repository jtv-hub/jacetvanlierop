"""Trade Sync Verifier — checks that trades.log matches trades.db."""

from __future__ import annotations

from crypto_trading_bot.db.sqlite_adapter import SQLiteAdapter
from crypto_trading_bot.ledger.trade_ledger import TradeLedger


class DummyPositionManager:
    """Minimal stub so TradeLedger can be instantiated in isolation."""

    def __init__(self) -> None:
        self.positions = {}


KEYS_TO_CHECK = [
    "status",
    "entry_price",
    "exit_price",
    "roi",
    "pnl_usd",
]


def _normalize_file_trade(trade: dict) -> dict:
    norm = dict(trade)
    # status: executed -> open for DB comparison
    if norm.get("status") == "executed":
        norm["status"] = "open"
    # side: long/short -> buy/sell
    side = norm.get("side")
    if isinstance(side, str):
        s = side.strip().lower()
        if s == "long":
            norm["side"] = "buy"
        elif s == "short":
            norm["side"] = "sell"
    # Map realized_gain -> pnl_usd for comparison
    if norm.get("pnl_usd") is None and norm.get("realized_gain") is not None:
        norm["pnl_usd"] = norm.get("realized_gain")
    return norm


def verify_trade_sync() -> None:
    """Compare JSONL trades vs. SQLite rows and print any mismatches."""

    ledger = TradeLedger(DummyPositionManager())
    sqlite_adapter = SQLiteAdapter()

    trades = list(ledger.trades or [])
    if not trades:
        ledger.reload_trades()
        trades = list(ledger.trades or [])

    keys_to_check = KEYS_TO_CHECK
    mismatches = 0
    checked = 0

    print("\n🔍 Verifying trade sync between trades.log and trades.db...\n")

    for trade in trades:
        trade_id = trade.get("trade_id")
        if not trade_id:
            continue

        checked += 1
        db_trade = sqlite_adapter.fetch_one(
            "SELECT * FROM trades WHERE trade_id = ?",
            (trade_id,),
        )
        if not db_trade:
            print(f"❌ Trade {trade_id} missing from SQLite DB.")
            mismatches += 1
            continue

        ftrade = _normalize_file_trade(trade)
        mismatch_key = None
        for key in keys_to_check:
            file_val = ftrade.get(key)
            db_val = db_trade.get(key)
            if file_val != db_val:
                mismatch_key = key
                print(
                    f"❌ Mismatch for {trade_id}: {key} file={file_val} db={db_val}",
                )
                mismatches += 1
                break
        if mismatch_key is None:
            print(f"✅ Trade {trade_id} is in sync.")

    if checked == 0:
        print("⚠️ No trades available to compare.")
        return

    if mismatches == 0:
        print(f"\n✅ {checked} trades checked, all matched.")
    else:
        print(f"\n❌ {mismatches} mismatches out of {checked} trades.")


def main() -> None:
    """Entry point when executed as a script."""

    verify_trade_sync()


if __name__ == "__main__":
    main()
