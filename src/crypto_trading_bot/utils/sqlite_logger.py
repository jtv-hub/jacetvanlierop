"""
SQLite logging utilities for the crypto trading bot.

This module replaces the previous JSONL append-only logs with three SQLite
databases stored under ``/db``:

- ``trades.db`` for executed trades
- ``positions.db`` for open position snapshots
- ``learning_feedback.db`` for learning machine diagnostics

It provides thread-safe logging helpers, schema-checked storage, and migration
helpers to import existing JSONL data. Running the module directly executes the
migration workflow and prints a concise summary.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from .file_locks import _locked_file
except ImportError:  # pragma: no cover - script execution fallback
    from crypto_trading_bot.utils.file_locks import _locked_file  # type: ignore


def init_db() -> None:
    """Ensure the confidence_log table exists."""

    Path("data").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect("data/learning.db")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS confidence_log (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                type TEXT,
                strategy TEXT,
                confidence REAL,
                regime TEXT,
                status TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
DB_DIR = BASE_DIR / "db"
LOGS_DIR = BASE_DIR / "logs"

TRADES_LOG_PATH = LOGS_DIR / "trades.log"
POSITIONS_LOG_PATH = LOGS_DIR / "positions.jsonl"
LEARNING_FEEDBACK_LOG_PATHS = [
    LOGS_DIR / "learning_feedback.log",
    LOGS_DIR / "learning_feedback.jsonl",
]

DB_DIR.mkdir(parents=True, exist_ok=True)


def process_trade_confidence(
    *,
    strategy: str,
    confidence: float,
    regime: str = "unknown",
    status: str = "recorded",
) -> None:
    """Persist confidence updates to the dedicated SQLite confidence log."""

    Path("data").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect("data/learning.db")
    try:
        conn.execute(
            """
            INSERT INTO confidence_log (timestamp, type, strategy, confidence, regime, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                "confidence_update",
                strategy,
                float(confidence),
                regime,
                status,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _json_dumps(payload: dict) -> str:
    """Serialize ``payload`` using compact JSON for storage."""

    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def _normalize_timestamp(value: object) -> str:
    """Return an ISO-8601 timestamp, defaulting to ``datetime.now`` when absent."""

    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _safe_status(value: object, default: str = "unknown") -> str:
    """Return a lower-case status string with a sane fallback."""

    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return default


def _read_json_lines(path: Path) -> Iterator[dict]:
    """Yield JSON objects from ``path`` while holding a shared file lock."""

    if not path.exists():
        return
    try:
        with _locked_file(str(path), "r") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping invalid JSON in %s", path)
    except FileNotFoundError:
        return


@dataclass(frozen=True)
class _DatabaseConfig:
    filename: str
    schema: str


class _SQLiteStore:
    """Thin wrapper around :mod:`sqlite3` providing locking and schema setup."""

    def __init__(self, config: _DatabaseConfig):
        self.config = config
        self.path = DB_DIR / config.filename
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level="DEFERRED",
            check_same_thread=False,
        )
        self._configure()
        self._initialise_schema()

    def _configure(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA temp_store=MEMORY;")
            self._conn.commit()

    def _initialise_schema(self) -> None:
        with self._lock:
            self._conn.executescript(self.config.schema)
            self._conn.commit()
            if self.config.filename == "trades.db":
                cursor = self._conn.execute("PRAGMA table_info(trades);")
                columns = {row[1] for row in cursor.fetchall()}
                if "state_vector" not in columns:
                    self._conn.execute("ALTER TABLE trades ADD COLUMN state_vector TEXT;")
                    self._conn.commit()
            if self.config.filename == "learning_feedback.db":
                cursor = self._conn.execute("PRAGMA table_info(learning_feedback);")
                columns = {row[1] for row in cursor.fetchall()}
                alterations: list[str] = []
                if "strategy" not in columns:
                    alterations.append("ALTER TABLE learning_feedback ADD COLUMN strategy TEXT;")
                if "parameters" not in columns:
                    alterations.append("ALTER TABLE learning_feedback ADD COLUMN parameters TEXT;")
                if "accepted" not in columns:
                    alterations.append("ALTER TABLE learning_feedback ADD COLUMN accepted INTEGER DEFAULT 0;")
                for statement in alterations:
                    self._conn.execute(statement)
                if alterations:
                    self._conn.commit()

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(self, sql: str, params: Iterable[object]) -> None:
        with self._transaction() as conn:
            conn.execute(sql, tuple(params))

    def fetchall(self, sql: str, params: Iterable[object] = ()) -> list[tuple]:
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return rows


_TRADES_DB = _SQLiteStore(
    _DatabaseConfig(
        filename="trades.db",
        schema="""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK(length(status) > 0),
            timestamp TEXT NOT NULL CHECK(length(timestamp) >= 10),
            payload TEXT NOT NULL,
            state_vector TEXT,
            recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
            CHECK(json_valid(payload))
        );
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        """,
    )
)

_POSITIONS_DB = _SQLiteStore(
    _DatabaseConfig(
        filename="positions.db",
        schema="""
        CREATE TABLE IF NOT EXISTS positions (
            trade_id TEXT PRIMARY KEY,
            pair TEXT NOT NULL CHECK(length(pair) > 0),
            status TEXT NOT NULL CHECK(status IN ('open','closed','closing','unknown')),
            timestamp TEXT NOT NULL CHECK(length(timestamp) >= 10),
            payload TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
            CHECK(json_valid(payload))
        );
        CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
        """,
    )
)

_LEARNING_DB = _SQLiteStore(
    _DatabaseConfig(
        filename="learning_feedback.db",
        schema="""
        CREATE TABLE IF NOT EXISTS learning_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suggestion_id TEXT,
            status TEXT CHECK(length(status) > 0),
            timestamp TEXT NOT NULL CHECK(length(timestamp) >= 10),
            strategy TEXT,
            parameters TEXT,
            accepted INTEGER DEFAULT 0,
            payload TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
            CHECK(json_valid(payload))
        );
        CREATE INDEX IF NOT EXISTS idx_learning_feedback_suggestion
            ON learning_feedback(suggestion_id);
        """,
    )
)


def log_trade(trade: dict) -> None:
    """Persist ``trade`` into the trades database."""

    if not isinstance(trade, dict):
        raise TypeError("trade must be a dictionary")
    trade_id = str(trade.get("trade_id") or trade.get("id") or "").strip()
    if not trade_id:
        raise ValueError("trade must include a non-empty trade_id")
    status = _safe_status(trade.get("status"))
    timestamp = _normalize_timestamp(trade.get("timestamp"))
    payload = _json_dumps(trade)
    state_vector = trade.get("state_vector")
    if state_vector is not None:
        try:
            vector_json = json.dumps(list(state_vector), separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            vector_json = None
    else:
        vector_json = None
    _TRADES_DB.execute(
        """
        INSERT INTO trades (trade_id, status, timestamp, payload, state_vector)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(trade_id) DO UPDATE SET
            status=excluded.status,
            timestamp=excluded.timestamp,
            payload=excluded.payload,
            state_vector=excluded.state_vector,
            recorded_at=datetime('now');
        """,
        (trade_id, status, timestamp, payload, vector_json),
    )


def log_position(position: dict) -> None:
    """Persist ``position`` into the positions database.

    Positions marked with ``status='closed'`` prune the stored row to mirror the
    previous JSONL behaviour of removing closed trades from the active list.
    """

    if not isinstance(position, dict):
        raise TypeError("position must be a dictionary")
    trade_id = str(position.get("trade_id") or "").strip()
    if not trade_id:
        raise ValueError("position must include a non-empty trade_id")

    status = _safe_status(position.get("status"), default="open")
    if status == "closed":
        _POSITIONS_DB.execute("DELETE FROM positions WHERE trade_id = ?", (trade_id,))
        return

    pair = str(position.get("pair") or "").strip()
    if not pair:
        raise ValueError("position must include a non-empty pair")

    timestamp = _normalize_timestamp(position.get("timestamp"))
    payload = _json_dumps(position)
    _POSITIONS_DB.execute(
        """
        INSERT INTO positions (trade_id, pair, status, timestamp, payload)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(trade_id) DO UPDATE SET
            pair=excluded.pair,
            status=excluded.status,
            timestamp=excluded.timestamp,
            payload=excluded.payload,
            recorded_at=datetime('now');
        """,
        (trade_id, pair, status, timestamp, payload),
    )


def log_learning_feedback(feedback: dict) -> None:
    """Persist ``feedback`` emitted by the learning pipeline."""

    if not isinstance(feedback, dict):
        raise TypeError("feedback must be a dictionary")

    state_vector = feedback.get("state_vector")
    if state_vector is not None:
        if np is not None and isinstance(state_vector, np.ndarray):
            feedback["state_vector"] = state_vector.astype(float).tolist()
        elif not isinstance(state_vector, list):
            try:
                feedback["state_vector"] = list(state_vector)  # type: ignore[arg-type]
            except TypeError:
                feedback["state_vector"] = None

    timestamp = _normalize_timestamp(feedback.get("timestamp"))
    suggestion_id = feedback.get("suggestion_id")
    status = feedback.get("status")
    strategy = feedback.get("strategy")
    parameters = feedback.get("parameters")
    if parameters is not None and not isinstance(parameters, str):
        try:
            parameters = json.dumps(parameters, separators=(",", ":"))
        except (TypeError, ValueError):
            parameters = None
    accepted_value = 1 if bool(feedback.get("accepted")) else 0
    payload = _json_dumps({**feedback, "timestamp": timestamp})

    _LEARNING_DB.execute(
        """
        INSERT INTO learning_feedback (suggestion_id, status, timestamp, strategy, parameters, accepted, payload)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (suggestion_id, status, timestamp, strategy, parameters, accepted_value, payload),
    )


def migrate_trades(log_path: Path = TRADES_LOG_PATH) -> int:
    """Migrate trades from ``log_path`` into SQLite."""

    migrated = 0
    for trade in _read_json_lines(log_path):
        try:
            log_trade(trade)
            migrated += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to migrate trade entry: %s", exc)
    return migrated


def migrate_positions(log_path: Path = POSITIONS_LOG_PATH) -> int:
    """Migrate active positions from ``log_path`` into SQLite."""

    migrated = 0
    for position in _read_json_lines(log_path):
        try:
            log_position(position)
            migrated += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to migrate position entry: %s", exc)
    return migrated


def migrate_learning_feedback(paths: Optional[Iterable[Path]] = None) -> int:
    """Migrate learning feedback records into SQLite."""

    migrated = 0
    candidates = list(paths) if paths is not None else LEARNING_FEEDBACK_LOG_PATHS
    for path in candidates:
        if not path.exists():
            continue
        for entry in _read_json_lines(path):
            try:
                log_learning_feedback(entry)
                migrated += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Failed to migrate learning feedback entry: %s", exc)
        break  # Only migrate from the first existing file
    return migrated


def run_migration() -> Dict[str, int]:
    """Execute all migrations and return per-log counts."""

    counts = {
        "trades_migrated": migrate_trades(),
        "positions_migrated": migrate_positions(),
        "feedback_migrated": migrate_learning_feedback(),
    }
    return counts


def update_learning_feedback(suggestion_id: str, updates: Dict[str, object]) -> None:
    """Update an existing learning feedback entry by suggestion_id."""

    if not suggestion_id or not updates:
        return
    db_path = DB_DIR / "learning_feedback.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT payload FROM learning_feedback WHERE suggestion_id = ?",
            (suggestion_id,),
        ).fetchone()
        if row is None:
            return
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        payload.update(updates)
        timestamp_value = payload.get("timestamp") or updates.get("timestamp")
        normalized_timestamp = _normalize_timestamp(timestamp_value)
        payload["timestamp"] = normalized_timestamp
        payload_json = _json_dumps(payload)
        strategy = payload.get("strategy")
        parameters = payload.get("parameters")
        if parameters is not None and not isinstance(parameters, str):
            try:
                parameters = json.dumps(parameters, separators=(",", ":"))
            except (TypeError, ValueError):
                parameters = None
        accepted_value = 1 if bool(payload.get("accepted")) else 0
        conn.execute(
            """
            UPDATE learning_feedback
            SET payload = ?, timestamp = ?, strategy = ?, parameters = ?, accepted = ?
            WHERE suggestion_id = ?
            """,
            (payload_json, normalized_timestamp, strategy, parameters, accepted_value, suggestion_id),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_ppo_shadow_stats() -> Dict[str, float]:
    """Return counts/winrate for PPO shadow trades based on learning feedback."""

    stats = {"accepted": 0.0, "wins": 0.0, "winrate": 0.0}
    db_path = DB_DIR / "learning_feedback.db"
    if not db_path.exists():
        return stats
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT payload FROM learning_feedback").fetchall()
    finally:
        conn.close()
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        if payload.get("strategy") != "ppo_agent":
            continue
        if bool(payload.get("accepted")):
            stats["accepted"] += 1.0
            roi = payload.get("actual_roi")
            try:
                roi_val = float(roi)
            except (TypeError, ValueError):
                roi_val = 0.0
            if roi_val > 0:
                stats["wins"] += 1.0
    if stats["accepted"] > 0:
        stats["winrate"] = stats["wins"] / stats["accepted"]
    return stats


def fetch_trades(status: Optional[str] = None, descending: bool = False) -> List[dict]:
    """Return trades ordered by timestamp."""

    clauses: list[str] = []
    params: list[object] = []
    if status:
        clauses.append("status = ?")
        params.append(status.lower())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "DESC" if descending else "ASC"
    rows = _TRADES_DB.fetchall(
        f"""
        SELECT payload, state_vector
        FROM trades
        {where}
        ORDER BY timestamp {order}, recorded_at {order};
        """,
        params,
    )
    trades: List[dict] = []
    for payload, state_vector_json in rows:
        try:
            trade = json.loads(payload)
        except json.JSONDecodeError:
            LOGGER.warning("Stored trade payload is invalid JSON; skipping entry.")
            continue
        if "state_vector" not in trade and state_vector_json:
            try:
                trade["state_vector"] = json.loads(state_vector_json)
            except json.JSONDecodeError:
                LOGGER.warning("Stored state_vector column invalid JSON; ignoring.")
        trades.append(trade)
    return trades


def fetch_positions(status: Optional[str] = None) -> List[dict]:
    """Return positions ordered by timestamp (defaults to open positions only)."""

    clauses: list[str] = []
    params: list[object] = []
    if status:
        clauses.append("status = ?")
        params.append(status.lower())
    if not status:
        clauses.append("status != 'closed'")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = _POSITIONS_DB.fetchall(
        f"""
        SELECT payload
        FROM positions
        {where}
        ORDER BY timestamp ASC, recorded_at ASC;
        """,
        params,
    )
    positions: List[dict] = []
    for (payload,) in rows:
        try:
            position = json.loads(payload)
        except json.JSONDecodeError:
            LOGGER.warning("Stored position payload is invalid JSON; skipping entry.")
            continue
        positions.append(position)
    return positions


if __name__ == "__main__":
    migrated_counts = run_migration()
    print(f"Migration complete: {migrated_counts}")
