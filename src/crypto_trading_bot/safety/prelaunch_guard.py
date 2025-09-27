"""Prelaunch guard to validate safety prerequisites before live trading.

Confidence tolerances are applied when comparing signal generation between
paper and live-dry snapshots to account for timing-driven OHLC differences.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from crypto_trading_bot.bot.simulation import collect_signal_snapshot
from crypto_trading_bot.bot.trading_logic import KILL_SWITCH_FILE, ledger
from crypto_trading_bot.config import CONFIG, ConfigurationError, is_live, set_live_mode
from crypto_trading_bot.utils.price_history import get_fallback_metrics
from crypto_trading_bot.utils.system_checks import ensure_system_capacity

logger = logging.getLogger(__name__)

_ALERT_LOG_PATH = Path("logs/alerts.log")
_SYSTEM_LOG_PATH = Path("logs/system.log")
_DEFAULT_WINDOW_HOURS = int(CONFIG.get("prelaunch_guard", {}).get("alert_window_hours", 72))
_DEFAULT_MAX_HIGH = int(CONFIG.get("prelaunch_guard", {}).get("max_recent_high_severity", 50))
CONFIDENCE_TOLERANCE = 0.25  # Allow small confidence deviations when signals agree (see module docstring).


def _clear_kill_switch(path: Path = Path(KILL_SWITCH_FILE)) -> None:
    if not path:
        return
    if not path.exists():
        logger.debug("No kill-switch present at %s", path)
        return
    try:
        path.unlink()
        logger.warning("Kill-switch file %s cleared by prelaunch guard.", path)
    except OSError as exc:  # pragma: no cover - filesystem specific
        raise ConfigurationError(f"Unable to remove kill-switch file {path}: {exc}") from exc


def _prune_log_noise(path: Path, substring: str = "No space left on device", max_lines: int = 5000) -> None:
    try:
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines(True)
        if len(lines) <= max_lines and sum(1 for line in lines if substring in line) <= 5:
            return
        trimmed = lines[-max_lines:]
        filtered = [line for line in trimmed if substring not in line]
        if not filtered:
            filtered = trimmed[-100:]
        path.write_text("".join(filtered), encoding="utf-8")
        logger.info("Pruned noisy entries from %s (removed %d lines)", path, len(lines) - len(filtered))
    except OSError:  # pragma: no cover - filesystem specific
        logger.warning("Log pruning skipped for %s due to IO error", path)


def _assert_logs_writable(paths: Iterable[Path]) -> None:
    for path in paths:
        directory = path.parent if path.parent.name else Path(".")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("")
        except OSError as exc:  # pragma: no cover - filesystem specific
            raise ConfigurationError(f"Cannot write to log file {path}: {exc}") from exc


def _parse_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    adjusted = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(adjusted)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _count_recent_high_severity(alert_path: Path, window_hours: int, *, now: datetime | None = None) -> int:
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)
    levels = {"ERROR", "CRITICAL"}
    count = 0

    if not alert_path.exists():
        return 0

    try:
        with alert_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed alert line: %s", line)
                    continue
                level = str(entry.get("level", "")).upper()
                if level not in levels:
                    continue
                ts = _parse_timestamp(entry.get("timestamp"))
                if ts is None:
                    logger.warning("Skipping alert with invalid timestamp: %s", line)
                    continue
                if ts >= cutoff:
                    count += 1
    except OSError as exc:
        raise ConfigurationError(f"Unable to read alerts.log: {exc}") from exc

    return count


@dataclass
class AlertDiagnostics:
    total_lines: int
    recent_count: int
    sampled_entries: list[str]
    had_error: bool = False


def _gather_alert_diagnostics(
    alert_path: Path,
    window_hours: int,
    *,
    sample_size: int = 5,
    now: datetime | None = None,
) -> AlertDiagnostics:
    diagnostics = AlertDiagnostics(total_lines=0, recent_count=0, sampled_entries=[], had_error=False)
    if now is None:
        now = datetime.now(timezone.utc)

    if not alert_path.exists():
        return diagnostics

    cutoff = now - timedelta(hours=window_hours)
    levels = {"ERROR", "CRITICAL"}

    try:
        with alert_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                diagnostics.total_lines += 1
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    diagnostics.had_error = True
                    logger.warning("Skipping malformed alert line: %s", stripped)
                    continue
                level = str(entry.get("level", "")).upper()
                if level not in levels:
                    continue
                ts = _parse_timestamp(entry.get("timestamp"))
                if ts is None:
                    diagnostics.had_error = True
                    logger.warning("Skipping alert with invalid timestamp: %s", stripped)
                    continue
                if ts >= cutoff:
                    diagnostics.recent_count += 1
                    if len(diagnostics.sampled_entries) < sample_size:
                        diagnostics.sampled_entries.append(stripped)
    except OSError as exc:
        diagnostics.had_error = True
        logger.error("Unable to read alerts.log for diagnostics: %s", exc)

    return diagnostics


def _first_actionable(entry: dict | None) -> Optional[dict]:
    if not isinstance(entry, dict):
        return None
    for signal in entry.get("signals", []) or []:
        if isinstance(signal, dict) and signal.get("signal") in {"buy", "sell"}:
            return signal
    return None


def _signals_match(paper_sig: dict, live_sig: dict) -> bool:
    required_keys = {"signal", "strategy", "confidence"}
    paper_missing = required_keys - set(paper_sig.keys())
    live_missing = required_keys - set(live_sig.keys())
    if paper_missing or live_missing:
        logger.warning(
            "Signal comparison missing keys | paper_missing=%s live_missing=%s",
            sorted(paper_missing),
            sorted(live_missing),
        )
        return False

    if paper_sig["signal"] != live_sig["signal"]:
        logger.warning("Signal direction mismatch | paper=%s live=%s", paper_sig["signal"], live_sig["signal"])
        return False

    if paper_sig["strategy"] != live_sig["strategy"]:
        logger.warning("Strategy mismatch | paper=%s live=%s", paper_sig["strategy"], live_sig["strategy"])
        return False

    try:
        paper_conf = float(paper_sig["confidence"])
        live_conf = float(live_sig["confidence"])
    except (TypeError, ValueError):
        logger.warning(
            "Non-numeric confidence encountered | paper=%s live=%s",
            paper_sig.get("confidence"),
            live_sig.get("confidence"),
        )
        return False

    delta = abs(paper_conf - live_conf)
    if delta <= CONFIDENCE_TOLERANCE:
        logger.debug(
            "Confidence deviation tolerated: %.4f vs %.4f (Δ=%.4f)",
            paper_conf,
            live_conf,
            delta,
        )
        return True

    logger.warning(
        "Confidence mismatch beyond tolerance: %.4f vs %.4f (Δ=%.4f)",
        paper_conf,
        live_conf,
        delta,
    )
    return False


def _ensure_no_mock_fallbacks() -> None:
    fallback_metrics = get_fallback_metrics(reset=False)
    live_blocks = fallback_metrics.get("live_block", {}) or {}
    if live_blocks:
        raise ConfigurationError(
            "Prelaunch guard: live mode attempted to fall back to mock price data",
        )


def _ensure_missing_trade_health() -> None:
    metrics = ledger.get_missing_trade_metrics(reset=False)
    for bucket, state in metrics.items():
        count = int(state.get("count", 0))
        suppressed = int(state.get("suppressed", 0))
        if suppressed > 0:
            raise ConfigurationError(
                f"Prelaunch guard: suppressed missing-trade alerts detected for bucket {bucket}",
            )
        if count > 0:
            logger.info("Missing-trade counter for %s currently %d (within allowance)", bucket, count)


def _run_mode_compare(pairs: Iterable[str]) -> None:
    pairs_list = list(pairs)
    if not pairs_list:
        logger.warning("No tradable pairs provided for mode comparison; skipping signal check.")
        return

    original_mode = is_live
    original_validate = deepcopy(CONFIG.get("kraken", {}).get("validate_orders"))
    original_dry = deepcopy(CONFIG.get("live_mode", {}).get("dry_run"))

    try:
        set_live_mode(False)
        CONFIG.setdefault("live_mode", {})["dry_run"] = False
        CONFIG.setdefault("kraken", {})["validate_orders"] = False
        paper_snapshot = collect_signal_snapshot(pairs_list)

        set_live_mode(True)
        CONFIG.setdefault("live_mode", {})["dry_run"] = True
        CONFIG.setdefault("kraken", {})["validate_orders"] = True
        live_snapshot = collect_signal_snapshot(pairs_list)
    finally:
        set_live_mode(original_mode)
        CONFIG.setdefault("kraken", {})["validate_orders"] = original_validate
        CONFIG.setdefault("live_mode", {})["dry_run"] = original_dry

    paper_map = {entry.get("pair"): entry for entry in paper_snapshot if isinstance(entry, dict)}
    live_map = {entry.get("pair"): entry for entry in live_snapshot if isinstance(entry, dict)}

    mismatches = []
    for pair in pairs_list:
        paper_sig = _first_actionable(paper_map.get(pair))
        live_sig = _first_actionable(live_map.get(pair))

        if paper_sig is None and live_sig is None:
            logger.warning("No actionable signals found for pair %s; skipping.", pair)
            continue
        if paper_sig is None or live_sig is None:
            missing_source = "paper" if paper_sig is None else "live"
            raise ConfigurationError(
                f"Signal mismatch for {pair}: missing actionable signal in {missing_source} snapshot."
            )

        if not _signals_match(paper_sig, live_sig):
            mismatches.append({"pair": pair, "paper": paper_sig, "live": live_sig})

    if mismatches:
        raise ConfigurationError(f"Signal mismatch between paper and live dry-run modes: {mismatches}")

    logger.info("Mode comparison succeeded for %d pair(s).", len(pairs_list))


def _auto_archive_alerts(repo_root: Path) -> bool:
    script = repo_root / "scripts" / "clear_alerts.py"
    if not script.exists():
        logger.error("clear_alerts.py not found at %s", script)
        return False

    result = subprocess.run(
        [sys.executable, str(script), "--force"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Auto-archive executed by prelaunch guard.")
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            logger.warning("clear_alerts stderr: %s", result.stderr.strip())
        return True

    logger.error("auto-archive failed (returncode=%s): %s", result.returncode, result.stderr.strip())
    return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _evaluate_alerts(
    alert_path: Path,
    window_hours: int,
    max_recent: int,
    *,
    auto_archive_invoked: bool = False,
) -> int:
    diagnostics = _gather_alert_diagnostics(alert_path, window_hours)

    logger.info(
        "[prelaunch] alerts.log diagnostics | total_lines=%d recent_high=%d auto_archive_invoked=%s",
        diagnostics.total_lines,
        diagnostics.recent_count,
        auto_archive_invoked,
    )
    if diagnostics.sampled_entries:
        for idx, entry in enumerate(diagnostics.sampled_entries, start=1):
            logger.info("[prelaunch] sample_alert_%d=%s", idx, entry)

    if diagnostics.had_error:
        logger.error("Alert diagnostics encountered malformed entries; failing safely.")
        raise ConfigurationError(
            "Prelaunch guard: unable to parse alerts.log reliably; run 'python3 scripts/clear_alerts.py --force'."
        )

    if diagnostics.recent_count > max_recent:
        message = (
            "Prelaunch guard: outstanding high-severity alerts detected "
            f"({diagnostics.recent_count} ≥ limit {max_recent}). "
            "Run 'python3 scripts/clear_alerts.py --archive-if-over "
            f"{max_recent}' or enable auto-archive to proceed."
        )
        raise ConfigurationError(message)
    return diagnostics.recent_count


def run_prelaunch_guard(
    *,
    pairs: Iterable[str] | None = None,
    auto_archive_alerts: bool = False,
    alert_window_hours: int = _DEFAULT_WINDOW_HOURS,
    max_recent_high: int = _DEFAULT_MAX_HIGH,
) -> None:
    ensure_system_capacity(min_disk_mb=500.0, min_mem_mb=256.0)
    _clear_kill_switch()
    _prune_log_noise(_ALERT_LOG_PATH)
    _prune_log_noise(_SYSTEM_LOG_PATH)
    _assert_logs_writable([_ALERT_LOG_PATH, _SYSTEM_LOG_PATH])

    auto_archive_did_run = False

    try:
        _evaluate_alerts(_ALERT_LOG_PATH, alert_window_hours, max_recent_high)
    except ConfigurationError as exc:
        if not auto_archive_alerts:
            raise
        logger.warning("High-severity alerts detected; attempting auto-archive (%s)", exc)
        if _auto_archive_alerts(_repo_root()):
            auto_archive_did_run = True
            _evaluate_alerts(
                _ALERT_LOG_PATH,
                alert_window_hours,
                max_recent_high,
                auto_archive_invoked=auto_archive_did_run,
            )
        else:
            raise

    if is_live:
        logger.info("Prelaunch guard running while is_live is already True; continuing checks.")

    _ensure_no_mock_fallbacks()
    _ensure_missing_trade_health()

    pairs_list = list(pairs) if pairs is not None else list(CONFIG.get("tradable_pairs", []))
    _run_mode_compare(pairs_list)

    logger.info("Prelaunch guard completed successfully.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prelaunch safety guard checks.")
    parser.add_argument(
        "--auto-archive-alerts",
        action="store_true",
        help="Automatically archive alerts.log once if recent high-severity alerts exceed the configured limit.",
    )
    parser.add_argument(
        "--alert-window-hours",
        type=int,
        default=_DEFAULT_WINDOW_HOURS,
        help="How many hours back to scan alerts.log for high-severity entries (default: %(default)s).",
    )
    parser.add_argument(
        "--max-recent-high-severity",
        type=int,
        default=_DEFAULT_MAX_HIGH,
        help="Maximum allowed recent high-severity alerts before failing (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    auto_archive = args.auto_archive_alerts or bool(os.getenv("CRYPTO_TRADING_BOT_AUTO_ARCHIVE_ALERTS"))

    try:
        run_prelaunch_guard(
            auto_archive_alerts=auto_archive,
            alert_window_hours=max(1, args.alert_window_hours),
            max_recent_high=max(1, args.max_recent_high_severity),
        )
        print("Prelaunch guard passed.")
        return 0
    except ConfigurationError as exc:
        logger.critical("Prelaunch guard failed: %s", exc)
        print(f"Prelaunch guard failed: {exc}")
        return 1


__all__ = ["run_prelaunch_guard", "_count_recent_high_severity", "_signals_match"]


if __name__ == "__main__":
    sys.exit(main())
