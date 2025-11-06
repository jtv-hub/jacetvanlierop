"""Standalone smoke test to verify TradeLedger dual-write sync with SQLite."""

import uuid
from datetime import datetime, timezone

from crypto_trading_bot.db.sqlite_adapter import SQLiteAdapter
from crypto_trading_bot.ledger.trade_ledger import TradeLedger

TRADE_ID = str(uuid.uuid4())


class DummyPositionManager:
    """Minimal stub so TradeLedger can satisfy its position_manager dependency."""

    def __init__(self):
        self.positions = {}


def main() -> None:
    """Create, update, and read back a trade to verify SQLite dual-write."""
    position_manager = DummyPositionManager()
    ledger = TradeLedger(position_manager)
    sqlite_adapter = SQLiteAdapter()

    trade_timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Create and log a trade
    ledger.log_trade(
        trading_pair="BTC/USDC",
        trade_size=500,
        strategy_name="test_strategy",
        side="buy",
        trade_id=TRADE_ID,
        timestamp=trade_timestamp,
        entry_price=42000.0,
        confidence=0.8,
        reason="test_entry",
    )
    print("✅ Logged trade.")

    # 2. Verify entry state in SQLite (status='open')
    fetched_open = sqlite_adapter.fetch_one(
        "SELECT * FROM trades WHERE trade_id = ?",
        (TRADE_ID,),
    )
    assert fetched_open is not None, "❌ Entry not found in SQLite after log_trade()."
    assert fetched_open["status"] == "open", f"❌ Expected status 'open', got {fetched_open['status']}"

    # 3. Close the trade (minimal kwargs)
    ledger.update_trade(
        TRADE_ID,
        exit_price=42350.0,
        reason="test_exit",
        exit_reason="manual_test",
    )
    print("✅ Updated trade.")

    # 4. Fetch from SQLite (post-update)
    fetched = sqlite_adapter.fetch_one(
        "SELECT * FROM trades WHERE trade_id = ?",
        (TRADE_ID,),
    )

    print("\n✅ Trade found in SQLite DB:")
    print(fetched)

    # 5. Assert that trade was written and closed
    assert fetched is not None, "❌ Trade not found in SQLite DB."
    assert fetched["trade_id"] == TRADE_ID, "❌ Fetched trade_id mismatch."
    assert fetched["status"] == "closed", f"❌ Expected status 'closed', got {fetched['status']}"
    assert fetched["exit_price"] is not None, "❌ exit_price not stored."
    assert fetched["roi"] is not None, "❌ roi not stored."
    assert fetched["pnl_usd"] is not None, "❌ pnl_usd not stored."
    print("🎉 Test passed: SQLite dual-write verified!")


if __name__ == "__main__":
    main()
