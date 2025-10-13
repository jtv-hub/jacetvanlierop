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
TIME_TOLERANCE_SECONDS = 30 * 60  # tolerate +/- 30 minutes between entries
VOLUME_TOLERANCE = 1e-6  # absolute volume tolerance for matching
SWITCH_ENV_VAR = "USDC_SWITCH_TIMESTAMP"

REVERSE_PAIR_MAP = {v: k for k, v in PAIR_MAP.items()}


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
        log_trade = LogTrade(
            trade_id=str(item.get("trade_id") or ""),
            timestamp=ts,
            pair=pair,
            normalized_pair=normalized_pair,
            size=float(size),
            side=(item.get("side") or "").lower(),
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

    for log_trade in log_trades:
        best_idx: Optional[int] = None
        best_delta: float = float("inf")
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

    report: Dict[str, Any] = {}

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

    matches: List[Tuple[LogTrade, KrakenTrade, float]] = []
    missing_logs: List[LogTrade] = []
    missing_kraken: List[KrakenTrade] = []
    if kraken_trades:
        matches, missing_logs, missing_kraken = match_trades(log_trades, kraken_trades)

    report["matched_trades"] = len(matches)
    report["unmatched_log_trades"] = len(missing_logs) if kraken_trades else len(log_trades)
    report["unmatched_kraken_trades"] = len(missing_kraken)

    if matches:
        avg_time_delta = sum(delta for *_, delta in matches) / len(matches)
        report["avg_match_time_delta_seconds"] = round(avg_time_delta, 2)
    if missing_logs:
        report["log_unmatched_samples"] = [trade.raw for trade in missing_logs[:5]]
    if missing_kraken:
        report["kraken_unmatched_samples"] = [trade.raw for trade in missing_kraken[:5]]

    if kraken_trades:
        quote_counts = Counter(trade.pair.split("/")[-1] for trade in kraken_trades)
        report["kraken_quote_usage"] = dict(quote_counts)

    internal_state = load_internal_portfolio_state()
    report["internal_portfolio_state"] = internal_state

    balance_difference: Optional[float] = None
    if balances and internal_state:
        internal_usdc = float(internal_state.get("available_capital") or 0.0)
        kraken_usdc = balances.get("USDC") or balances.get("USD") or balances.get("ZUSD")
        if kraken_usdc is not None:
            balance_difference = kraken_usdc - internal_usdc
    report["balance_difference"] = balance_difference

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
        summary_lines.append(f"Balance difference (Kraken - internal): {balance_difference:.6f}")
    else:
        summary_lines.append("Balance difference (Kraken - internal): n/a")
    if report["usd_pair_violations"]:
        violations = report["usd_pair_violations"]
        summary_lines.append(f"USD pair violations after switch: {violations}")
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
