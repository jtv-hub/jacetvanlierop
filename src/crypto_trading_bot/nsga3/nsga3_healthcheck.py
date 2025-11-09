"""
nsga3_healthcheck.py — Verifies environment safety and data integrity before
launching a new NSGA-III optimization cycle.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/nsga3_alerts.log")
os.makedirs(LOG_PATH.parent, exist_ok=True)
SCHEDULER_LOG_PATH = Path("logs/nsga3_scheduler.log")
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)

try:
    from crypto_trading_bot.risk import risk_guard as risk_guard_module
except ImportError:  # pragma: no cover - optional dependency
    LOGGER.warning("HealthCheck: unable to import risk_guard module; skipping.")
    risk_guard_module = None

RiskGuard = getattr(risk_guard_module, "RiskGuard", None) if risk_guard_module else None


def _project_root() -> Path:
    """Resolve project root based on this file location (package-aware)."""
    # nsga3_healthcheck.py -> nsga3 -> crypto_trading_bot -> src -> project root
    return Path(__file__).resolve().parents[3]


def _resolve_db_path() -> Path:
    """Return absolute path to the trades.db honoring NSGA3_DB_PATH env override."""
    override = os.environ.get("NSGA3_DB_PATH")
    if override:
        p = Path(override).expanduser().resolve()
        return p
    root = _project_root()
    return (root / "db" / "trades.db").resolve()


def _ensure_schema(conn: sqlite3.Connection) -> bool:
    """Create the trades table if missing; return True on success."""
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                pair TEXT,
                roi REAL DEFAULT 0.0,
                drawdown REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                regime TEXT,
                params TEXT,
                notes TEXT
            );
            """
        )
        conn.commit()
        try:
            # Attempt durability; some platforms may ignore fsync for SQLite WAL.
            conn.execute("PRAGMA wal_checkpoint(FULL);")
        except sqlite3.Error:
            pass
        return True
    except sqlite3.Error as exc:
        LOGGER.exception("HealthCheck: schema ensure failed: %s", exc)
        return False


def _log_scheduler_diagnostic(event: str) -> None:
    """Append a one-line diagnostic to the scheduler log."""
    SCHEDULER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event}
    with SCHEDULER_LOG_PATH.open("a", encoding="utf-8") as handle:
        import json as _json

        _json.dump(entry, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _check_database() -> bool:
    """Verify that the trades database exists and holds sufficient rows.

    Applies idempotent migration to ensure schema before counting rows. Only
    fails if DB is inaccessible or migrations cannot be applied.
    """
    db_path = _resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as conn:
            # Migration step: ensure schema
            created = _ensure_schema(conn)
            if created:
                _log_scheduler_diagnostic("db_migration:applied")
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM trades;")
            count = int(cur.fetchone()[0])
    except sqlite3.Error as exc:
        LOGGER.exception("HealthCheck: database error: %s", exc)
        return False

    if count < 100:
        LOGGER.warning("HealthCheck: insufficient trade data (<100 rows).")
        return False
    return True


def _check_logs() -> bool:
    """Ensure that critical NSGA-III log artifacts exist."""
    monitored = ("nsga3_checkpoint.json", "nsga3_promotions.jsonl")
    for filename in monitored:
        if _log_exists(filename):
            continue
        LOGGER.warning("HealthCheck: missing log file %s", filename)
        return False
    return True


def _log_exists(filename: str) -> bool:
    """Return True if either legacy or regime-specific file exists."""
    base_path = Path("logs", filename)
    if base_path.exists():
        return True
    stem, suffix = os.path.splitext(filename)
    if suffix:
        candidate = Path("logs", f"{stem}_global{suffix}")
        if candidate.exists():
            return True
    return False


def _check_risk_guard() -> bool:
    """
    Validate that the global RiskGuard is present and not paused.

    If the module is unavailable in this environment, log a warning and
    continue so that developers can still run dry-runs locally.
    """
    if RiskGuard is None:
        LOGGER.warning("HealthCheck: RiskGuard module unavailable; skipping pause check.")
        return True
    try:
        guard = RiskGuard()
    except (
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:  # pragma: no cover - constructor failure
        LOGGER.exception("HealthCheck: RiskGuard init failed: %s", exc)
        return False
    if guard.is_paused():
        LOGGER.warning("HealthCheck: RiskGuard is PAUSED — drawdown cap hit.")
        return False
    return True


def run_healthcheck() -> bool:
    """Run all health checks; return True if environment is safe to proceed."""
    db_ok = _check_database()
    log_ok = _check_logs()
    risk_ok = _check_risk_guard()
    healthy = db_ok and log_ok and risk_ok
    LOGGER.info("%s", "HealthCheck: OK" if healthy else "HealthCheck: FAILED.")
    return healthy


if __name__ == "__main__":
    print("PASS" if run_healthcheck() else "FAIL")
