"""Reconcile local trade logs with Kraken account activity.

This script performs the following checks:
- Queries Kraken private endpoints (TradesHistory, Balance).
- Loads recent trades from logs/trades.log and attempts to match them to
  Kraken trades by pair, size, and timestamp proximity.
- Flags trades present in logs but missing from Kraken (and vice versa).
- Validates that all Kraken trade pairs use USDC as the quote currency.
- Compares Kraken balances with the internal portfolio state snapshot.
- Scans daemon logs for live trading activity, confirming USDC usage.

Run with:
    PYTHONPATH=src python3 scripts/audit_kraken_reconciliation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from crypto_trading_bot.config import IS_LIVE
from crypto_trading_bot.ledger.trade_ledger import TradeLedger
from crypto_trading_bot.utils.kraken_api import kraken_client
from crypto_trading_bot.utils.kraken_client import KrakenAPIError
from crypto_trading_bot.utils.kraken_pairs import PAIR_MAP, ensure_usdc_pair

LOG_PATH = Path("logs/trades.log")
PORTFOLIO_STATE_PATH = Path("data/portfolio_state.json")
DAEMON_LOG_PATH = Path("logs/daemon.out")
DAEMON_ARCHIVE_DIR = Path("logs/archive")
DAEMON_RETENTION_HOURS = 48
DAEMON_CRITICAL_WINDOW_HOURS = 24
TRADES_TO_CONSIDER = 200  # limit reconciliation to the most recent trades
try:
    _TIME_TOLERANCE_RAW = float(os.getenv("RECONCILIATION_TIME_TOLERANCE_SECONDS", "5"))
except (TypeError, ValueError):
    _TIME_TOLERANCE_RAW = 5.0
TIME_TOLERANCE_SECONDS = max(0, int(round(_TIME_TOLERANCE_RAW)))
VOLUME_TOLERANCE = float(os.getenv("RECONCILIATION_VOLUME_TOLERANCE", "0.0001"))
SWITCH_ENV_VAR = "USDC_SWITCH_TIMESTAMP"
BALANCE_TOLERANCE = float(os.getenv("KRAKEN_BALANCE_TOLERANCE", "0.01"))
RECONCILIATION_GRACE_SECONDS = int(os.getenv("RECONCILIATION_GRACE_SECONDS", "60"))
LEGACY_MISMATCH_HOURS = float(os.getenv("RECONCILIATION_LEGACY_HOURS", "24"))
RECONCILE_IMPORT_ENABLED = os.getenv("RECONCILE_IMPORT_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
}

REVERSE_PAIR_MAP = {v: k for k, v in PAIR_MAP.items()}


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


DRY_RUN_MODE = _truthy(os.getenv("FINAL_AUDIT_DRY_RUN"))
IGNORE_UNMATCHED_PAPER_TRADES = _truthy(os.getenv("AUDIT_IGNORE_UNMATCHED_PAPER_TRADES", "1"))

logger = logging.getLogger(__name__)


class _LedgerPositionStub:
    """Minimal stub to satisfy TradeLedger's position manager dependency."""

    def __init__(self) -> None:
        self.positions: Dict[str, Any] = {}


def _extract_daemon_timestamp(line: str) -> Optional[datetime]:
    """Best-effort extraction of a timestamp from a daemon log line."""

    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[.,]\d+)?", line)
    if not match:
        return None
    raw = match.group(1)
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)


def _cleanup_daemon_log(
    *,
    path: Path = DAEMON_LOG_PATH,
    archive_dir: Path = DAEMON_ARCHIVE_DIR,
    retention_hours: int = DAEMON_RETENTION_HOURS,
    critical_window_hours: int = DAEMON_CRITICAL_WINDOW_HOURS,
) -> Dict[str, Any]:
    """Purge stale entries from daemon.out or archive when no recent CRITICAL alerts exist."""

    result: Dict[str, Any] = {"cleaned": False, "archived": False}
    if not path.exists():
        result["reason"] = "missing"
        return result

    now = datetime.now(timezone.utc)
    retention_cutoff = now - timedelta(hours=max(retention_hours, 1))
    critical_cutoff = now - timedelta(hours=max(critical_window_hours, 1))

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except OSError as exc:
        result["reason"] = f"read_error:{exc}"
        return result

    last_critical: Optional[datetime] = None
    parsed_lines: list[tuple[Optional[datetime], str]] = []
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        ts = _extract_daemon_timestamp(line)
        parsed_lines.append((ts, line))
        if "CRITICAL" in line.upper() and ts is not None:
            if last_critical is None or ts > last_critical:
                last_critical = ts

    if last_critical is None or last_critical <= critical_cutoff:
        # No recent critical alerts — archive entire log and reset.
        if lines:
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"daemon_{now:%Y%m%d%H%M%S}.out"
            try:
                shutil.move(str(path), archive_path)
                path.write_text("", encoding="utf-8")
                result.update({"archived": True, "archive_path": str(archive_path)})
            except OSError as exc:
                result["reason"] = f"archive_error:{exc}"
                return result
        result["cleaned"] = True
        result["last_critical"] = last_critical.isoformat() if last_critical else None
        return result

    # Retain recent entries, drop anything older than retention window.
    kept_lines: list[str] = []
    for ts, line in parsed_lines:
        if ts is None:
            continue
        if ts >= retention_cutoff:
            kept_lines.append(line)

    if len(kept_lines) != len(lines):
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"daemon_{now:%Y%m%d%H%M%S}.out"
        stale_lines = [line for ts, line in parsed_lines if ts is None or ts < retention_cutoff]
        try:
            with archive_path.open("w", encoding="utf-8") as handle:
                for entry in stale_lines:
                    handle.write(entry + "\n")
            with path.open("w", encoding="utf-8") as handle:
                for entry in kept_lines:
                    handle.write(entry + "\n")
            result.update(
                {
                    "cleaned": True,
                    "archived": True,
                    "archive_path": str(archive_path),
                    "pruned_count": len(stale_lines),
                }
            )
        except OSError as exc:
            result["reason"] = f"prune_error:{exc}"
            return result
    else:
        result["reason"] = "no_prune_needed"
    result["last_critical"] = last_critical.isoformat()
    return result


def _coerce_utc(value: datetime) -> datetime:
    """Return ``value`` normalized to a timezone-aware UTC datetime."""

    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def detect_switch_time_from_logs() -> Optional[datetime]:
    """Infer the first USDC timestamp in the local trade log, if present."""

    if not LOG_PATH.exists():
        return None
    earliest: Optional[datetime] = None
    with open(LOG_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            pair = (entry.get("pair") or "").upper().replace("-", "/")
            if not pair.endswith("/USDC"):
                continue
            ts = _parse_iso(entry.get("timestamp"))
            if ts is None:
                continue
            if earliest is None or ts < earliest:
                earliest = ts
    return earliest


def resolve_switch_time(cli_value: Optional[str]) -> Optional[datetime]:
    """Resolve the USD→USDC switch timestamp from CLI, env, or auto detection."""

    raw = cli_value.strip() if cli_value else None
    if not raw:
        raw = os.getenv(SWITCH_ENV_VAR, "").strip()
    if not raw:
        raw = "auto"
    if raw.lower() == "auto":
        return detect_switch_time_from_logs()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        message = "⚠️ Unable to parse switch timestamp " f"'{raw}'; falling back to auto detection."
        print(message, file=sys.stderr)
        return detect_switch_time_from_logs()
    return _coerce_utc(parsed)


@dataclass
class LogTrade:
    """Normalized representation of a trade entry from the local log."""

    trade_id: str
    timestamp: datetime
    pair: str
    normalized_pair: str
    size: float
    side: str
    txids: List[str]
    reconciled: bool
    pending_reconciliation: bool
    pending_reconciliation_at: Optional[datetime]
    source: str
    raw: Dict[str, Any]


@dataclass
class KrakenTrade:
    """Normalized representation of a trade entry returned by Kraken."""

    txid: str
    timestamp: datetime
    pair: str
    normalized_pair: str
    volume: float
    side: str
    raw: Dict[str, Any]


def _describe_log_trade(trade: LogTrade) -> Dict[str, Any]:
    return {
        "trade_id": trade.trade_id,
        "timestamp": trade.timestamp.isoformat(),
        "pair": trade.pair,
        "normalized_pair": trade.normalized_pair,
        "size": trade.size,
        "side": trade.side,
        "txids": trade.txids,
        "source": trade.source,
    }


def _describe_kraken_trade(trade: KrakenTrade) -> Dict[str, Any]:
    return {
        "txid": trade.txid,
        "timestamp": trade.timestamp.isoformat(),
        "pair": trade.pair,
        "normalized_pair": trade.normalized_pair,
        "volume": trade.volume,
        "side": trade.side,
    }


def _parse_iso(ts: str | None) -> Optional[datetime]:
    """Parse ISO timestamps (including ``Z`` suffix) into UTC datetimes."""

    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _kraken_time(value: Any) -> Optional[datetime]:
    """Convert Kraken numeric timestamps into UTC datetimes."""

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(seconds) and seconds > 0:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    return None


def _analyse_log_mismatch(
    trade: LogTrade,
    kraken_trades: Iterable[KrakenTrade],
    *,
    time_tolerance: int,
    volume_tolerance: float,
) -> Dict[str, Any]:
    """Return structured diagnostics describing why ``trade`` failed to match."""

    time_tolerance_seconds = max(0, int(time_tolerance))
    context: Dict[str, Any] = {
        "candidate_count": 0,
        "finite_candidate_count": 0,
        "time_tolerance_seconds": time_tolerance_seconds,
        "volume_tolerance": volume_tolerance,
        "effective_volume_tolerance": None,
        "best_volume_gap": None,
        "closest_timestamp_delta": None,
        "log_side": trade.side,
        "candidate_sides": [],
        "reason": "unclassified_mismatch",
    }

    candidates = [kt for kt in kraken_trades if kt.normalized_pair == trade.normalized_pair]
    context["candidate_count"] = len(candidates)
    if not candidates:
        context["reason"] = "no_remote_trades_for_pair"
        return context

    finite_candidates = [kt for kt in candidates if math.isfinite(kt.volume)]
    context["finite_candidate_count"] = len(finite_candidates)
    if not finite_candidates:
        context["reason"] = "remote_volume_missing"
        return context

    effective_volume_tolerance = max(volume_tolerance, 0.001 * trade.size)
    context["effective_volume_tolerance"] = effective_volume_tolerance

    volume_gaps = [abs(kt.volume - trade.size) for kt in finite_candidates]
    if volume_gaps:
        best_volume_gap = min(volume_gaps)
        context["best_volume_gap"] = best_volume_gap
        if best_volume_gap > effective_volume_tolerance:
            context["reason"] = f"volume_gap>{effective_volume_tolerance:.6f}"
            return context

    candidate_sides = {_kraken_side_to_position(kt.side) for kt in finite_candidates}
    context["candidate_sides"] = sorted(candidate_sides)
    if trade.side in {"long", "short"} and trade.side not in candidate_sides:
        context["reason"] = "side_mismatch"
        return context

    deltas = [abs((kt.timestamp - trade.timestamp).total_seconds()) for kt in finite_candidates]
    if deltas:
        best_delta = min(deltas)
        context["closest_timestamp_delta"] = best_delta
        if best_delta > time_tolerance_seconds:
            context["reason"] = f"timestamp_delta>{time_tolerance_seconds}s (min={best_delta:.1f}s)"
            return context

    return context


def _diagnose_log_mismatch(
    trade: LogTrade,
    kraken_trades: Iterable[KrakenTrade],
    *,
    time_tolerance: int,
    volume_tolerance: float,
) -> str:
    result = _analyse_log_mismatch(
        trade,
        kraken_trades,
        time_tolerance=time_tolerance,
        volume_tolerance=volume_tolerance,
    )
    return result.get("reason", "unclassified_mismatch")


def _candidate_summaries(
    log_trade: LogTrade,
    kraken_trades: Iterable[KrakenTrade],
    limit: int = 3,
) -> list[Dict[str, Any]]:
    candidates: list[tuple[float, KrakenTrade]] = []
    for remote in kraken_trades:
        if remote.normalized_pair != log_trade.normalized_pair:
            continue
        if not math.isfinite(remote.volume):
            continue
        delta = abs((remote.timestamp - log_trade.timestamp).total_seconds())
        candidates.append((delta, remote))
    candidates.sort(key=lambda item: item[0])
    summaries: list[Dict[str, Any]] = []
    for _, remote in candidates[:limit]:
        entry = _describe_kraken_trade(remote)
        entry["position_side"] = _kraken_side_to_position(remote.side)
        summaries.append(entry)
    return summaries


def _log_unmatched_trades(
    missing_logs: Iterable[LogTrade],
    missing_kraken: Iterable[KrakenTrade],
    kraken_trades: Iterable[KrakenTrade],
    *,
    time_tolerance: int,
    volume_tolerance: float,
) -> None:
    missing_logs_list = list(missing_logs)
    missing_kraken_list = list(missing_kraken)

    if missing_logs_list:
        logger.warning("Unmatched local trades detected: %d", len(missing_logs_list))
    for log_trade in missing_logs_list:
        analysis = _analyse_log_mismatch(
            log_trade,
            kraken_trades,
            time_tolerance=time_tolerance,
            volume_tolerance=volume_tolerance,
        )
        candidates = _candidate_summaries(log_trade, kraken_trades)
        context = {
            "trade": _describe_log_trade(log_trade),
            "analysis": analysis,
            "candidate_matches": candidates,
        }
        logger.warning("Unmatched local trade | %s", json.dumps(context, default=str))

    if missing_kraken_list:
        logger.warning("Unmatched Kraken trades detected: %d", len(missing_kraken_list))
    for remote in missing_kraken_list:
        context = {"trade": _describe_kraken_trade(remote)}
        logger.warning("Unmatched Kraken trade | %s", json.dumps(context, default=str))


def _partition_legacy_unmatched(log_trades: Iterable[LogTrade]) -> tuple[list[LogTrade], list[LogTrade]]:
    legacy: list[LogTrade] = []
    active: list[LogTrade] = []
    now = datetime.now(timezone.utc)
    for trade in log_trades:
        age_hours = max(0.0, (now - trade.timestamp).total_seconds() / 3600)
        if age_hours >= LEGACY_MISMATCH_HOURS:
            legacy.append(trade)
        else:
            active.append(trade)
    return legacy, active


def _cleanup_asset(code: str) -> str:
    """Normalize Kraken asset codes (strip prefixes, harmonise aliases)."""

    token = (code or "").upper()
    token = token.replace("ZUSD", "USD")
    token = token.replace("XBT", "BTC")
    token = token.replace("XXBT", "BTC")
    token = token.replace("XETH", "ETH")
    while token.startswith(("X", "Z")) and len(token) > 1:
        token = token[1:]
    return token


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_trade_ledger() -> TradeLedger:
    """Instantiate a TradeLedger bound to the default log paths."""

    return TradeLedger(_LedgerPositionStub())


def balance_difference_within_tolerance(
    kraken_balances: Dict[str, float],
    internal_state: Dict[str, Any],
    tolerance: float,
) -> Tuple[Optional[float], bool]:
    """Return balance delta (Kraken - internal) and whether it fits tolerance."""

    if not kraken_balances or not internal_state:
        return None, True

    internal_usdc = _safe_float(internal_state.get("available_capital")) or 0.0
    kraken_usdc = (
        _safe_float(kraken_balances.get("USDC"))
        or _safe_float(kraken_balances.get("USD"))
        or _safe_float(kraken_balances.get("ZUSD"))
    )
    if kraken_usdc is None:
        return None, True

    diff = kraken_usdc - internal_usdc
    return diff, abs(diff) <= tolerance


def kraken_pair_to_human(raw_pair: str | None) -> str:
    """Translate Kraken pair tokens into ``BASE/QUOTE`` strings."""

    if not raw_pair:
        return "UNKNOWN/UNKNOWN"
    pair_token = raw_pair.replace("-", "").replace("/", "").upper()
    if pair_token in REVERSE_PAIR_MAP:
        return REVERSE_PAIR_MAP[pair_token]

    if "/" in raw_pair:
        base, quote = raw_pair.split("/", 1)
        return f"{_cleanup_asset(base)}/{_cleanup_asset(quote)}"

    # Fallback: attempt to split evenly (Kraken often prefixes with X/Z per leg)
    length = len(pair_token)
    if length in (6, 8):
        half = length // 2
        base = pair_token[:half]
        quote = pair_token[half:]
    else:
        # default to last 3 chars as quote
        base = pair_token[:-3]
        quote = pair_token[-3:]
    return f"{_cleanup_asset(base)}/{_cleanup_asset(quote)}"


def to_usdc_pair(pair: str) -> str:
    """Ensure that the provided pair string uses a USDC quote."""

    return ensure_usdc_pair(pair)


def load_log_trades(
    limit: int = TRADES_TO_CONSIDER,
    *,
    switch_time: Optional[datetime] = None,
) -> Tuple[List[LogTrade], List[LogTrade]]:
    """Load closed trades from the local log, filtering after ``switch_time``."""

    if not LOG_PATH.exists():
        return [], []

    buffer: deque[Dict[str, Any]] = deque(maxlen=limit)
    with open(LOG_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (entry.get("status") or "").lower() != "closed":
                continue
            buffer.append(entry)

    trades: List[LogTrade] = []
    usd_after_switch: List[LogTrade] = []
    for item in buffer:
        ts = _parse_iso(item.get("timestamp"))
        if ts is None:
            continue
        pair = (item.get("pair") or "").upper().replace("-", "/")
        size = item.get("size")
        if not isinstance(size, (int, float)):
            continue
        if switch_time and ts < switch_time:
            continue
        try:
            normalized_pair = to_usdc_pair(pair)
        except ValueError:
            normalized_pair = pair
        raw_txid = item.get("txid")
        if isinstance(raw_txid, list):
            txids = [str(val) for val in raw_txid if val]
        elif isinstance(raw_txid, str) and raw_txid:
            txids = [raw_txid]
        else:
            txids = []
        reconciled_flag = bool(item.get("reconciled", False))
        pending_flag = bool(item.get("pending_reconciliation", False))
        source_flag = item.get("source", "local")
        pending_at_raw = item.get("pending_reconciliation_at")
        pending_at = _parse_iso(pending_at_raw) if pending_at_raw else None

        log_trade = LogTrade(
            trade_id=str(item.get("trade_id") or ""),
            timestamp=ts,
            pair=pair,
            normalized_pair=normalized_pair,
            size=float(size),
            side=_normalize_log_side(item.get("side")),
            txids=txids,
            reconciled=reconciled_flag,
            pending_reconciliation=pending_flag,
            pending_reconciliation_at=pending_at,
            source=source_flag,
            raw=item,
        )
        quote = normalized_pair.split("/", 1)[-1] if "/" in normalized_pair else ""
        if quote != "USDC":
            usd_after_switch.append(log_trade)
        trades.append(log_trade)
    return trades, usd_after_switch


def load_kraken_trades(
    *,
    switch_time: Optional[datetime] = None,
) -> Tuple[List[KrakenTrade], Dict[str, Any]]:
    """Fetch private Kraken trades applying the optional switch-time filter."""

    payload = kraken_client.query_private(
        "TradesHistory",
        {"trades": True, "type": "all"},
        return_result=False,
    )
    result = payload.get("result") or {}
    trades_blob = result.get("trades") or {}
    trades: List[KrakenTrade] = []
    for txid, data in trades_blob.items():
        pair = kraken_pair_to_human(data.get("pair"))
        normalized_pair = to_usdc_pair(pair)
        time_value = _kraken_time(data.get("time") or data.get("closetm"))
        if time_value is None:
            continue
        if switch_time and time_value < switch_time:
            continue
        try:
            volume = float(data.get("vol") or data.get("volum") or data.get("volume"))
        except (TypeError, ValueError):
            volume = math.nan
        side = (data.get("type") or "").lower()
        trades.append(
            KrakenTrade(
                txid=str(txid),
                timestamp=time_value,
                pair=pair,
                normalized_pair=normalized_pair,
                volume=volume,
                side=side,
                raw=data,
            ),
        )
    trades.sort(key=lambda t: t.timestamp)
    return trades, payload


def load_kraken_balance() -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return the sanitized Kraken balance payload mapping assets to floats."""

    payload = kraken_client.query_private("Balance", {}, return_result=False)
    raw_balances = payload.get("result") or {}
    normalized: Dict[str, float] = {}
    for code, value in raw_balances.items():
        human = _cleanup_asset(code)
        try:
            normalized[human] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized, payload


def _kraken_side_to_position(side: str) -> str:
    side_normalized = (side or "").strip().lower()
    if side_normalized == "sell":
        return "short"
    if side_normalized == "buy":
        return "long"
    return side_normalized or "unknown"


def _normalize_log_side(side: Any) -> str:
    """Normalize local trade side labels to ``long``/``short``."""

    value = (str(side) if side is not None else "").strip().lower()
    if value in {"long", "buy"}:
        return "long"
    if value in {"short", "sell"}:
        return "short"
    return value


def match_trades(
    log_trades: Iterable[LogTrade],
    kraken_trades: Iterable[KrakenTrade],
    *,
    time_tolerance: int = TIME_TOLERANCE_SECONDS,
    volume_tolerance: float = VOLUME_TOLERANCE,
) -> Tuple[List[Tuple[LogTrade, KrakenTrade, float]], List[LogTrade], List[KrakenTrade]]:
    """Best-effort trade matching between log entries and Kraken executions."""

    kraken_pool: List[KrakenTrade] = list(kraken_trades)
    matched_indices: set[int] = set()
    matches: List[Tuple[LogTrade, KrakenTrade, float]] = []
    log_missing: List[LogTrade] = []

    txid_lookup: Dict[str, int] = {trade.txid: idx for idx, trade in enumerate(kraken_pool)}

    for log_trade in log_trades:
        best_idx: Optional[int] = None
        best_delta: float = float("inf")

        # Prefer matching by txid when available.
        for tx in log_trade.txids:
            idx = txid_lookup.get(tx)
            if idx is not None and idx not in matched_indices:
                best_idx = idx
                best_delta = 0.0
                break

        if best_idx is None:
            for idx, candidate in enumerate(kraken_pool):
                if idx in matched_indices:
                    continue
                if candidate.normalized_pair != log_trade.normalized_pair:
                    continue
                if not math.isfinite(candidate.volume):
                    continue
                volume_gap = abs(candidate.volume - log_trade.size)
                tolerance = max(volume_tolerance, 0.001 * log_trade.size)
                if volume_gap > tolerance:
                    continue
                candidate_side = _kraken_side_to_position(candidate.side)
                if log_trade.side and log_trade.side in {"long", "short"}:
                    if candidate_side and candidate_side != log_trade.side:
                        continue
                delta = abs((candidate.timestamp - log_trade.timestamp).total_seconds())
                if delta <= time_tolerance and delta < best_delta:
                    best_idx = idx
                    best_delta = delta

        if best_idx is not None:
            matched_indices.add(best_idx)
            matches.append((log_trade, kraken_pool[best_idx], best_delta))
        else:
            log_missing.append(log_trade)

    kraken_missing = [trade for idx, trade in enumerate(kraken_pool) if idx not in matched_indices]
    return matches, log_missing, kraken_missing


def reconcile_trades(
    *,
    matches: Iterable[Tuple[LogTrade, KrakenTrade, float]],
    missing_log_trades: Iterable[LogTrade],
    missing_kraken_trades: Iterable[KrakenTrade],
    ledger: TradeLedger,
    grace_seconds: int = RECONCILIATION_GRACE_SECONDS,
) -> Dict[str, Any]:
    """Import missing Kraken trades and mark unmatched log trades as pending."""

    imported = 0
    pending_ids: List[str] = []
    reconciled_ids: List[str] = []

    now = datetime.now(timezone.utc)

    for log_trade, _kraken_trade, _ in matches:
        ledger.mark_reconciled(log_trade.trade_id)
        reconciled_ids.append(log_trade.trade_id)

    for log_trade in missing_log_trades:
        age = (now - log_trade.timestamp).total_seconds()
        already_pending = log_trade.pending_reconciliation
        if not already_pending:
            already_pending = log_trade.pending_reconciliation_at is not None
        if age >= grace_seconds and not already_pending:
            ledger.mark_pending_reconciliation(log_trade.trade_id, pending=True, timestamp=now)
            pending_ids.append(log_trade.trade_id)

    for kraken_trade in missing_kraken_trades:
        if ledger.find_trade_by_txid(kraken_trade.txid):
            continue

        if not RECONCILE_IMPORT_ENABLED:
            continue

        raw = kraken_trade.raw
        cost_val = _safe_float(raw.get("cost"))
        fee_val = _safe_float(raw.get("fee")) or 0.0
        price = _safe_float(raw.get("price")) or _safe_float(raw.get("avg_price"))
        if price is None and cost_val is not None and kraken_trade.volume:
            price = cost_val / max(kraken_trade.volume, 1e-12)
        if price is None or price <= 0:
            price = 1.0
        if cost_val is None:
            cost_val = abs(price * kraken_trade.volume)

        gross = abs(cost_val)
        if kraken_trade.side == "sell":
            net_amount = gross - abs(fee_val)
        else:
            net_amount = -(gross + abs(fee_val))

        fills = [
            {
                "price": price,
                "quantity": kraken_trade.volume,
                "cost": gross,
                "fee": abs(fee_val),
                "type": raw.get("type") or kraken_trade.side,
                "time": kraken_trade.timestamp.isoformat(),
            }
        ]

        trade_id = raw.get("ordertxid") or f"kraken-{kraken_trade.txid}"
        ledger.log_trade(
            trading_pair=kraken_trade.pair,
            trade_size=kraken_trade.volume,
            strategy_name="ReconciliationImport",
            trade_id=trade_id,
            confidence=1.0,
            entry_price=price or 0.0,
            txid=[kraken_trade.txid],
            fills=fills,
            gross_amount=gross,
            fee=abs(fee_val),
            net_amount=net_amount,
            balance_delta=net_amount,
            reconciled=True,
            pending_reconciliation=False,
            source="kraken_import",
        )
        ledger.mark_reconciled(trade_id)
        imported += 1

    return {
        "kraken_trades_imported": imported,
        "log_trades_marked_pending": len(pending_ids),
        "pending_trade_ids": pending_ids,
        "reconciled_trade_ids": reconciled_ids,
        "import_enabled": RECONCILE_IMPORT_ENABLED,
    }


def load_internal_portfolio_state() -> Dict[str, Any]:
    """Load the latest internal portfolio snapshot if it exists."""

    if PORTFOLIO_STATE_PATH.exists():
        with open(PORTFOLIO_STATE_PATH, "r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return {}
    return {}


def inspect_daemon_logs(limit: int = 1000) -> Dict[str, Any]:
    """Inspect daemon log tail for live-trade activity and drawdown lines."""

    maintenance = _cleanup_daemon_log()
    if not DAEMON_LOG_PATH.exists():
        return {"found": False, "lines": [], "maintenance": maintenance}

    pattern_live = re.compile(r"Submitting live trade .*pair=(?P<pair>[^\s|]+)")

    buffer: deque[str] = deque(maxlen=limit)
    with open(DAEMON_LOG_PATH, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            buffer.append(line.strip())

    live_trades: List[str] = []
    usd_pairs: List[str] = []
    drawdown_lines: List[str] = []
    for line in buffer:
        live_match = pattern_live.search(line)
        if live_match:
            pair = live_match.group("pair")
            live_trades.append(line)
            if pair.upper().endswith("USD") and not pair.upper().endswith("USDC"):
                usd_pairs.append(line)
        if "[DRAWDOWN]" in line:
            drawdown_lines.append(line)
    return {
        "found": True,
        "live_trade_lines": live_trades[-5:],
        "usd_quote_lines": usd_pairs[-5:],
        "drawdown_lines": drawdown_lines[-5:],
        "maintenance": maintenance,
    }


def safe_run(fn, *args, **kwargs):
    """Execute ``fn`` and capture Kraken/network related failures uniformly."""

    try:
        return {"status": "ok", "data": fn(*args, **kwargs)}
    except KrakenAPIError as exc:
        return {"status": "error", "error": str(exc)}
    except (ValueError, RuntimeError, OSError, TypeError) as exc:  # pragma: no cover
        return {"status": "error", "error": f"unexpected error: {exc}"}


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point for the reconciliation audit."""

    parser = argparse.ArgumentParser(description="Reconcile local trades with Kraken account data")
    parser.add_argument(
        "--switch-time",
        dest="switch_time",
        help=(
            "ISO-8601 timestamp for the USD→USDC transition (default comes from "
            f"env {SWITCH_ENV_VAR!s} or auto-detect)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=TRADES_TO_CONSIDER,
        help="Number of most recent trades to inspect from the local log",
    )
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {"warnings": []}

    switch_time = resolve_switch_time(args.switch_time)
    report["switch_timestamp"] = switch_time.isoformat() if switch_time else None

    log_trades, usd_after_switch = load_log_trades(limit=args.limit, switch_time=switch_time)
    report["log_trade_count"] = len(log_trades)
    report["usd_pair_violations"] = len(usd_after_switch)
    if usd_after_switch:
        report["usd_pair_samples"] = [trade.raw for trade in usd_after_switch[:5]]

    trades_result = safe_run(load_kraken_trades, switch_time=switch_time)
    report["kraken_trade_status"] = trades_result["status"]
    if trades_result["status"] == "ok":
        kraken_trades, raw_payload = trades_result["data"]
        report["kraken_trade_count"] = len(kraken_trades)
        report["kraken_payload_error"] = raw_payload.get("error")
    else:
        kraken_trades = []
        report["kraken_trade_count"] = 0
        report["kraken_trade_error"] = trades_result.get("error")

    balance_result = safe_run(load_kraken_balance)
    report["kraken_balance_status"] = balance_result["status"]
    if balance_result["status"] == "ok":
        balances, balance_payload = balance_result["data"]
        report["kraken_balances"] = balances
        report["kraken_balance_error"] = balance_payload.get("error")
    else:
        balances = {}
        report["kraken_balances"] = {}
        report["kraken_balance_error"] = balance_result.get("error")

    ledger = load_trade_ledger()

    matches: List[Tuple[LogTrade, KrakenTrade, float]] = []
    missing_logs: List[LogTrade] = []
    missing_kraken: List[KrakenTrade] = []
    if kraken_trades:
        matches, missing_logs, missing_kraken = match_trades(log_trades, kraken_trades)

    report["matched_trades"] = len(matches)
    legacy_unmatched: list[LogTrade] = []
    active_unmatched: list[LogTrade] = []
    if missing_logs:
        legacy_unmatched, active_unmatched = _partition_legacy_unmatched(missing_logs)
    missing_logs = active_unmatched

    report["legacy_unmatched_log_trades"] = len(legacy_unmatched)
    if legacy_unmatched:
        report["legacy_unmatched_details"] = [_describe_log_trade(trade) for trade in legacy_unmatched]

    report["unmatched_log_trades"] = len(missing_logs) if kraken_trades else len(log_trades)
    report["unmatched_kraken_trades"] = len(missing_kraken)
    ignore_unmatched_logs = IGNORE_UNMATCHED_PAPER_TRADES and (DRY_RUN_MODE or not IS_LIVE)

    if missing_logs or missing_kraken:
        _log_unmatched_trades(
            missing_logs,
            missing_kraken,
            kraken_trades,
            time_tolerance=TIME_TOLERANCE_SECONDS,
            volume_tolerance=VOLUME_TOLERANCE,
        )
        unmatched_details: list[Dict[str, Any]] = []
        for log_trade in missing_logs:
            analysis = _analyse_log_mismatch(
                log_trade,
                kraken_trades,
                time_tolerance=TIME_TOLERANCE_SECONDS,
                volume_tolerance=VOLUME_TOLERANCE,
            )
            entry = _describe_log_trade(log_trade)
            entry["reason"] = analysis.get("reason", "unclassified_mismatch")
            entry["analysis"] = analysis
            unmatched_details.append(entry)
        report["unmatched_log_details"] = unmatched_details
        report["unmatched_kraken_details"] = [_describe_kraken_trade(remote) for remote in missing_kraken]

    if missing_logs:
        mode_label = "dry-run" if DRY_RUN_MODE else ("paper" if not IS_LIVE else "live")
        trade_summary = {
            "type": "unmatched_log_trades",
            "count": len(missing_logs),
            "ignored": bool(ignore_unmatched_logs),
            "mode": mode_label,
            "trade_ids": [trade.trade_id for trade in missing_logs if trade.trade_id],
        }
        report.setdefault("warnings_json", []).append(trade_summary)
        report["unmatched_log_warning_summary"] = trade_summary
        if ignore_unmatched_logs:
            structured_message = f"Unmatched paper trades ignored ({mode_label} mode)"
            logger.warning("%s | %s", structured_message, json.dumps(trade_summary, default=str))
            report["warnings"].append(structured_message)

    reconciliation_details: Dict[str, Any] = {}
    if matches or missing_logs or missing_kraken:
        reconciliation_details = reconcile_trades(
            matches=matches,
            missing_log_trades=missing_logs,
            missing_kraken_trades=missing_kraken,
            ledger=ledger,
        )
        report.update(reconciliation_details)
        if reconciliation_details.get("log_trades_marked_pending"):
            report["warnings"].append(
                f"Marked {reconciliation_details['log_trades_marked_pending']} trade(s) pending reconciliation."
            )
        if reconciliation_details.get("kraken_trades_imported"):
            report["warnings"].append(
                f"Imported {reconciliation_details['kraken_trades_imported']} trade(s) from Kraken history."
            )

    if matches:
        avg_time_delta = sum(delta for *_, delta in matches) / len(matches)
        report["avg_match_time_delta_seconds"] = round(avg_time_delta, 2)
    if missing_logs:
        report["log_unmatched_samples"] = [trade.raw for trade in missing_logs[:5]]
    if missing_kraken:
        report["kraken_unmatched_samples"] = [trade.raw for trade in missing_kraken[:5]]

    if reconciliation_details:
        pending_count = reconciliation_details.get("log_trades_marked_pending", 0)
        imported_count = reconciliation_details.get("kraken_trades_imported", 0)
        if pending_count or imported_count:
            report["reconciliation_status"] = "warning" if IS_LIVE else "error"
        else:
            report["reconciliation_status"] = "ok"
    else:
        report["reconciliation_status"] = "ok"

    if kraken_trades:
        quote_counts = Counter(trade.pair.split("/")[-1] for trade in kraken_trades)
        report["kraken_quote_usage"] = dict(quote_counts)

    internal_state = load_internal_portfolio_state()
    report["internal_portfolio_state"] = internal_state

    balance_difference, within_tol = balance_difference_within_tolerance(
        balances,
        internal_state,
        BALANCE_TOLERANCE,
    )
    report["balance_difference"] = balance_difference
    report["balance_within_tolerance"] = within_tol
    if balance_difference is not None and not within_tol:
        report["warnings"].append(f"Balance difference {balance_difference:.2f} exceeds ±{BALANCE_TOLERANCE:.2f} USD.")

    logs_inspection = inspect_daemon_logs()
    report["daemon_log_inspection"] = logs_inspection

    kraken_status = report["kraken_trade_status"]
    summary_lines = [
        "=== Kraken Reconciliation Summary ===",
        f"Switch timestamp: {report['switch_timestamp'] or 'auto (not found)'}",
        f"Log trades considered: {report['log_trade_count']}",
        ("Kraken trades fetched: " f"{report['kraken_trade_count']} (status={kraken_status})"),
        f"Matched trades: {report['matched_trades']}",
        f"Unmatched log trades: {report['unmatched_log_trades']}",
        f"Unmatched Kraken trades: {report['unmatched_kraken_trades']}",
    ]
    if balance_difference is not None:
        status = "within" if within_tol else "exceeds"
        summary_lines.append(
            f"Balance difference (Kraken - internal): {balance_difference:.6f} ({status} ±{BALANCE_TOLERANCE:.2f})"
        )
    else:
        summary_lines.append("Balance difference (Kraken - internal): n/a")
    if report["usd_pair_violations"]:
        violations = report["usd_pair_violations"]
        summary_lines.append(f"USD pair violations after switch: {violations}")
    if reconciliation_details.get("kraken_trades_imported"):
        summary_lines.append(
            f"Imported {reconciliation_details['kraken_trades_imported']} trade(s) from Kraken history."
        )
    if reconciliation_details.get("log_trades_marked_pending"):
        summary_lines.append(f"Pending reconciliation trades: {reconciliation_details['log_trades_marked_pending']}")
    if report["warnings"]:
        summary_lines.append("Warnings: " + "; ".join(report["warnings"]))
    print("\n".join(summary_lines))

    print()
    print(json.dumps(report, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    """JSON serializer hook for dataclasses and datetime objects."""

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (LogTrade, KrakenTrade)):
        return obj.__dict__
    return str(obj)


if __name__ == "__main__":
    main()
