"""
sqlite_adapter.py

SQLite database adapter for trade lifecycle tracking, RL episodes, and strategy evolution.
Provides singleton connection, optimized PRAGMAs, and batch/dual-write support.
"""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteAdapter:
    """
    Singleton adapter for interacting with the project's main SQLite database.
    Provides optimized write performance, WAL mode, foreign key enforcement,
    and batch operations for high‑speed backfill or RL replay logging.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        # Enforce singleton so all modules share the same DB connection
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if hasattr(self, "conn"):
            return  # Already initialized

        if db_path is None:
            base_dir = os.path.dirname(__file__)
            db_path = os.path.join(base_dir, "trades.db")
        self.db_path = db_path
        self.logger = logger

        # Thread‑safe mode enabled (required for scheduler, RL, and strategies)
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=15,
            isolation_level=None,  # autocommit
        )
        self.conn.row_factory = sqlite3.Row

        # Enforce durability + concurrency improvements
        self._optimize_pragmas()
        self._ensure_indexes()

    # ---------------------------------------------------------
    # PRAGMA optimization (WAL, cache tuning, FK enforcement)
    # ---------------------------------------------------------
    def _optimize_pragmas(self):
        try:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous = NORMAL;
                PRAGMA foreign_keys = ON;
                PRAGMA temp_store = MEMORY;
                PRAGMA cache_size = -64000;  -- 64MB page cache
                PRAGMA busy_timeout = 15000;
                """
            )
        except sqlite3.Error as e:
            print(f"[SQLite PRAGMA Error] {e}")

    # ---------------------------------------------------------
    # Index creation to speed PPO, NSGA‑3 & equity tracking
    # ---------------------------------------------------------
    def _ensure_indexes(self):
        try:
            self.conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_trades_pnl
                    ON trades(pnl_usd DESC, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_trades_strategy_month
                    ON trades(strategy_id, strftime('%Y-%m', timestamp));

                CREATE INDEX IF NOT EXISTS idx_rl_episode
                    ON rl_actions(episode_id, timestamp);

                CREATE INDEX IF NOT EXISTS idx_rl_reward
                    ON rl_actions(reward DESC);

                CREATE INDEX IF NOT EXISTS idx_nsga_projection
                    ON nsga_generations(monthly_roi_projected DESC);

                CREATE INDEX IF NOT EXISTS idx_nsga_gen_live
                    ON nsga_generations(gen DESC)
                    WHERE rank = 1;

                CREATE INDEX IF NOT EXISTS idx_equity_month
                    ON equity_curve(strftime('%Y-%m', timestamp));
                """
            )
        except sqlite3.Error as e:
            print(f"[SQLite Index Error] {e}")

    # ---------------------------------------------------------
    # Insert single trade (primary use: dual‑write from trade_ledger)
    # ---------------------------------------------------------
    def insert_trade(self, trade: Dict[str, Any]):
        """Insert or update a single trade row into the SQLite 'trades' table.
        Used during live diual-write logging from trade_ledger.py"""
        trade_obj = trade if isinstance(trade, dict) else {}
        trade_id = trade_obj.get("trade_id")
        trade_keys = list(trade_obj.keys())
        # Defensive defaults for nullable numerics
        for k in ("roi", "pnl_usd", "rsi", "slippage_bps", "exit_price"):
            if k in trade_obj and trade_obj[k] is None:
                pass
        logger.debug(
            "SQLite insert_trade start trade_id=%s keys=%s payload=%s",
            trade_id,
            trade_keys,
            trade_obj,
        )
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO trades (
                    trade_id, timestamp, pair, side, entry_price, exit_price,
                    size_usd, roi, pnl_usd, confidence, strategy_id,
                    status, exit_reason, rsi, slippage_bps
                )
                VALUES (
                    :trade_id, :timestamp, :pair, :side, :entry_price, :exit_price,
                    :size_usd, :roi, :pnl_usd, :confidence, :strategy_id,
                    :status, :exit_reason, :rsi, :slippage_bps
                );
                """,
                trade_obj,
            )
            logger.debug("SQLite insert_trade success trade_id=%s", trade_id)
        except sqlite3.Error as e:
            logger.error(
                "SQLite insert_trade failed trade_id=%s keys=%s error=%s payload=%s",
                trade_id,
                trade_keys,
                e,
                trade_obj,
            )
            print(f"[SQLite Error] insert_trade() failed: {e}")

    # ---------------------------------------------------------
    # Batch insert (used for backfill from JSONL)
    # ---------------------------------------------------------
    def batch_insert_trades(self, trades: List[Dict[str, Any]]):
        """Efficiently insert or update multiple trade rows using executemany().
        Used for migrations, backfills, and large RL replay imports"""
        count = len(trades or [])
        logger.info("SQLite batch insert start count=%s", count)
        try:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO trades (
                    trade_id, timestamp, pair, side, entry_price, exit_price,
                    size_usd, roi, pnl_usd, confidence, strategy_id,
                    status, exit_reason, rsi, slippage_bps
                )
                VALUES (
                    :trade_id, :timestamp, :pair, :side, :entry_price, :exit_price,
                    :size_usd, :roi, :pnl_usd, :confidence, :strategy_id,
                    :status, :exit_reason, :rsi, :slippage_bps
                );
                """,
                trades,
            )
            logger.debug("SQLite batch insert success count=%s", count)
        except sqlite3.Error as e:
            logger.warning("SQLite batch insert failed count=%s error=%s", count, e)
            print(f"[SQLite Error] batch_insert_trades() failed: {e}")

    # ---------------------------------------------------------
    # Generic query helpers
    # ---------------------------------------------------------
    def fetch_all(self, query: str, params: Optional[tuple] = None):
        """Execute a SELECT query and return all results as list of dicts."""
        try:
            cur = self.conn.execute(query, params or ())
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as e:
            print(f"[SQLite Error] fetch_all failed: {e}")
            return []

    def fetch_one(self, query: str, params: Optional[tuple] = None):
        """Execute a SECELT query and return a single row as a dictionary.
        Return None if no result is found."""
        try:
            cur = self.conn.execute(query, params or ())
            row = cur.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            print(f"[SQLite Error] fetch_one failed: {e}")
            return None

    def get_recent_trades(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent trades as lightweight dicts for shadow simulations.
        """
        trades: List[Dict[str, Any]] = []
        db_path = Path(self.db_path)
        if not db_path.exists():
            return trades

        query = "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?"
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query, (limit,)).fetchall()
                trades = [dict(row) for row in rows]
        # pylint: disable=broad-exception-caught
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.exception("SQLiteAdapter get_recent_trades failed: %s", exc)
        return trades

    # ---------------------------------------------------------
    # Close safely (rarely used; DB stays open for bot lifetime)
    # ---------------------------------------------------------
    def close(self):
        """Close the underlying SQLite connection safely."""
        try:
            self.conn.close()
        except sqlite3.Error as exc:
            logger.debug("SQLiteAdapter close() ignored error: %s", exc)
