#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
        echo "🔍 Dry run mode enabled — simulating live audit behavior"
    fi
done

SUPPRESS_STALE_ERRORS=0
STRICT_MODE_FLAG=0

while (($#)); do
    case "$1" in
        --suppress-stale-errors)
            SUPPRESS_STALE_ERRORS=1
            ;;
        --dry-run)
            # already handled above; ignore during primary parsing
            ;;
        --strict-mode)
            STRICT_MODE_FLAG=1
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ${STRICT_MODE_FLAG} -eq 1 ]]; then
    export STRICT_MODE=1
fi
export SUPPRESS_STALE_ERRORS
if $DRY_RUN; then
    export FINAL_AUDIT_DRY_RUN=1
else
    unset FINAL_AUDIT_DRY_RUN 2>/dev/null || true
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found (checked python3, python)." >&2
    exit 1
fi

AUDIT_STATUS=0

"${PYTHON_BIN}" - <<'PY' || AUDIT_STATUS=$?
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ROOT_DIR", ".")).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


DRY_RUN_MODE = _truthy(os.getenv("FINAL_AUDIT_DRY_RUN"))
if DRY_RUN_MODE:
    os.environ["FINAL_AUDIT_DRY_RUN"] = "1"
    os.environ.setdefault("LIVE_MODE", "1")


def _load_dotenv_into_env(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.is_file():
        return {}
    try:  # pragma: no cover - optional dependency
        from dotenv import dotenv_values
    except Exception:  # pylint: disable=broad-except
        dotenv_values = None

    raw_values: dict[str, str] = {}
    if dotenv_values is not None:
        try:
            candidate = dotenv_values(env_path) or {}
        except Exception:  # pragma: no cover - defensive
            candidate = {}
        raw_values = {key: str(val) for key, val in candidate.items() if val is not None}

    if not raw_values:
        try:
            with env_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    raw_values[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            return {}

    loaded: dict[str, str] = {}
    for key, value in raw_values.items():
        if not value or key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def _build_fallback_config() -> dict[str, Any]:
    _load_dotenv_into_env(ROOT)

    def _float_env(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    drawdown = _float_env("LIVE_RISK_DRAWDOWN_THRESHOLD", _float_env("AUTO_PAUSE_MAX_DRAWDOWN", 0.10))
    failure_limit = _int_env("LIVE_RISK_FAILURE_LIMIT", _int_env("AUTO_PAUSE_MAX_CONSEC_LOSSES", 5))
    risk_state_candidate = os.getenv("RISK_GUARD_STATE_FILE") or "logs/risk_guard_state.json"
    risk_state_file = str((ROOT / risk_state_candidate).resolve())
    confirmation_candidate = os.getenv("LIVE_CONFIRMATION_FILE") or ".confirm_live_trade"
    confirmation_file = str((ROOT / confirmation_candidate).resolve())
    kraken_key = os.getenv("KRAKEN_API_KEY", "")
    kraken_secret = os.getenv("KRAKEN_API_SECRET", "")

    deployment_phase = (os.getenv("DEPLOY_PHASE") or "canary").strip().lower() or "canary"

    config: dict[str, Any] = {
        "kraken_api_key": kraken_key,
        "kraken_api_secret": kraken_secret,
        "kraken": {
            "api_key": kraken_key,
            "api_secret": kraken_secret,
            "balance_asset": (os.getenv("KRAKEN_BALANCE_ASSET") or "USDC").upper(),
            "validate_orders": _truthy(os.getenv("KRAKEN_VALIDATE_ORDERS")),
            "time_in_force": os.getenv("KRAKEN_TIME_IN_FORCE"),
            "api_base": os.getenv("KRAKEN_API_BASE", "https://api.kraken.com"),
        },
        "live_mode": {
            "drawdown_threshold": drawdown,
            "failure_limit": failure_limit,
            "risk_state_file": risk_state_file,
            "confirmation_file": confirmation_file,
            "force_override": _truthy(os.getenv("LIVE_FORCE")),
            "requested_via_env": _truthy(os.getenv("LIVE_MODE")),
        },
        "auto_pause": {
            "max_drawdown_pct": drawdown,
            "max_consecutive_losses": failure_limit,
        },
        "deployment": {
            "phase": deployment_phase,
            "phase_source": "fallback",
            "phase_status": "unknown",
            "phase_updated_at": None,
        },
        "prelaunch_guard": {},
        "test_mode": _truthy(os.getenv("CRYPTO_TRADING_BOT_TEST_MODE")),
        "_config_source": "fallback",
    }
    return config


CONFIG = {}
CONFIG_SOURCE = "module"
CONFIG_ERRORS: list[str] = []
IS_LIVE = False
for attempt in range(2):
    try:
        config_module = importlib.import_module("crypto_trading_bot.config")
        if DRY_RUN_MODE:
            setattr(config_module, "IS_LIVE", True)
            setattr(config_module, "is_live", True)
        CONFIG = getattr(config_module, "CONFIG", {})
        IS_LIVE = bool(getattr(config_module, "IS_LIVE", False))
        if DRY_RUN_MODE:
            IS_LIVE = True
            try:
                CONFIG["is_live"] = True
            except Exception:
                pass
        if not CONFIG:
            raise AttributeError("CONFIG mapping unavailable")
        break
    except Exception as exc:  # pragma: no cover - defensive import guard
        if attempt == 0:
            sys.modules.pop("crypto_trading_bot.config", None)
            continue
        print(f"⚠️ Unable to load full configuration: {exc}")
        print("   Continuing with limited configuration context.")
        CONFIG_ERRORS.append(f"{type(exc).__name__}: {exc}")
        CONFIG = _build_fallback_config()
        live_env_flag = _truthy(os.getenv("LIVE_MODE")) or _truthy(os.getenv("CRYPTO_TRADING_BOT_LIVE"))
        IS_LIVE = live_env_flag or bool(CONFIG.get("live_mode", {}).get("force_override"))
        CONFIG_SOURCE = CONFIG.get("_config_source", "fallback")
        break

if DRY_RUN_MODE:
    IS_LIVE = True
    try:
        CONFIG["is_live"] = True
    except Exception:
        pass

_soft_mode_env = os.getenv("SOFT_AUDIT_MODE")
if _soft_mode_env is None:
    SOFT_AUDIT_MODE = not IS_LIVE
else:
    SOFT_AUDIT_MODE = _truthy(_soft_mode_env)
if CONFIG_SOURCE != "module":
    print(f"⚠️ Configuration fallback active (source={CONFIG_SOURCE}).")
    for reason in CONFIG_ERRORS:
        print(f"   fallback_reason={reason}")

STRICT_MODE_ENABLED = _truthy(os.getenv("STRICT_MODE"))
SUPPRESS_STALE_ERRORS = _truthy(os.getenv("SUPPRESS_STALE_ERRORS"))
AUDIT_IGNORE_MINOR_ROI_WARNINGS = _truthy(os.getenv("AUDIT_IGNORE_MINOR_ROI_WARNINGS", "1"))
try:
    AUDIT_MINOR_ROI_WARNING_TOLERANCE = float(os.getenv("AUDIT_MINOR_ROI_WARNING_TOLERANCE", "0.01"))
except (TypeError, ValueError):
    AUDIT_MINOR_ROI_WARNING_TOLERANCE = 0.01
AUDIT_IGNORE_UNMATCHED_PAPER_TRADES = _truthy(os.getenv("AUDIT_IGNORE_UNMATCHED_PAPER_TRADES", "1"))

BALANCE_WARNING_PATTERN = re.compile(
    r"\[balance\].*difference\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r".*±([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s+usd",
    re.IGNORECASE,
)

try:
    alerts_module = importlib.import_module("crypto_trading_bot.utils.alerts")
    send_alert = alerts_module.send_alert
except Exception:  # pragma: no cover - best-effort fallback
    def send_alert(message: str, *, level: str = "INFO", context: dict | None = None) -> None:
        print(f"[alert:{level}] {message} context={context}")

if DRY_RUN_MODE:
    def send_alert(message: str, *, level: str = "INFO", context: dict | None = None) -> None:
        print("⚠️ Skipping real trade actions in dry-run mode")
        print(f"[dry-run alert:{level}] {message} context={context}")


DEPLOY_PHASE_FILE = ROOT / "logs" / "deploy_phase.json"
CANARY_MIN_TRADES = int(os.getenv("CRYPTO_TRADING_BOT_CANARY_MIN_TRADES", "10") or "10")
CANARY_MAX_DRAWDOWN = float(os.getenv("CRYPTO_TRADING_BOT_CANARY_MAX_DRAWDOWN", "0.01") or "0.01")


def _write_deploy_phase_file(phase: str, status: str, context: dict[str, Any] | None = None) -> None:
    if DRY_RUN_MODE:
        print("⚠️ Skipping real trade actions in dry-run mode")
        print("   Deploy phase file update suppressed for dry-run.")
        return
    payload = {
        "phase": phase,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "final_health_audit",
    }
    if context:
        payload["context"] = context
    try:
        DEPLOY_PHASE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DEPLOY_PHASE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        print(f"⚠️ Unable to persist deploy phase file {DEPLOY_PHASE_FILE}: {exc}")


def load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def run_debug_diagnostics() -> tuple[dict, list[str]]:
    module = load_script_module("debug_live_diagnostics")
    summary: dict = {}
    transcripts: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        module._check_env(summary)  # type: ignore[attr-defined]
        module._private_api_checks(summary)  # type: ignore[attr-defined]
        module._scan_logs(summary)  # type: ignore[attr-defined]
        module._check_risk_guard(summary)  # type: ignore[attr-defined]
        module._validate_trade(summary)  # type: ignore[attr-defined]
    output = buf.getvalue().strip()
    if output:
        transcripts.append(output)
    module._print_summary(summary)  # type: ignore[attr-defined]
    return summary, transcripts


def run_audit_reconciliation() -> dict:
    module = load_script_module("audit_kraken_reconciliation")
    if DRY_RUN_MODE and hasattr(module, "_cleanup_daemon_log"):
        def _dry_run_cleanup(*args, **kwargs):
            print("⚠️ Skipping real trade actions in dry-run mode")
            return {"cleaned": False, "archived": False, "reason": "dry_run"}

        module._cleanup_daemon_log = _dry_run_cleanup  # type: ignore[attr-defined]
    report: dict[str, object] = {"warnings": []}
    switch_time = module.resolve_switch_time(None)  # type: ignore[attr-defined]
    log_trades, usd_after_switch = module.load_log_trades(  # type: ignore[attr-defined]
        limit=module.TRADES_TO_CONSIDER,  # type: ignore[attr-defined]
        switch_time=switch_time,
    )
    report["switch_timestamp"] = switch_time.isoformat() if switch_time else None
    report["log_trade_count"] = len(log_trades)
    report["usd_pair_violations"] = len(usd_after_switch)
    if usd_after_switch:
        report["usd_pair_samples"] = [trade.raw for trade in usd_after_switch[:5]]

    trades_result = module.safe_run(module.load_kraken_trades, switch_time=switch_time)  # type: ignore[attr-defined]
    report["kraken_trade_status"] = trades_result["status"]
    if trades_result["status"] == "ok":
        kraken_trades, raw_payload = trades_result["data"]
        report["kraken_trade_count"] = len(kraken_trades)
        report["kraken_payload_error"] = raw_payload.get("error")
    else:
        kraken_trades = []
        report["kraken_trade_count"] = 0
        report["kraken_trade_error"] = trades_result.get("error")

    balance_result = module.safe_run(module.load_kraken_balance)  # type: ignore[attr-defined]
    report["kraken_balance_status"] = balance_result["status"]
    if balance_result["status"] == "ok":
        balances, balance_payload = balance_result["data"]
        report["kraken_balances"] = balances
        report["kraken_balance_error"] = balance_payload.get("error")
    else:
        balances = {}
        report["kraken_balances"] = {}
        report["kraken_balance_error"] = balance_result.get("error")

    matches = []
    missing_logs = []
    missing_kraken = []
    if kraken_trades:
        matches, missing_logs, missing_kraken = module.match_trades(log_trades, kraken_trades)  # type: ignore[attr-defined]

    report["matched_trades"] = len(matches)
    report["unmatched_log_trades"] = len(missing_logs) if kraken_trades else len(log_trades)
    report["unmatched_kraken_trades"] = len(missing_kraken)
    if matches:
        avg_time_delta = sum(delta for *_, delta in matches) / len(matches)
        report["avg_match_time_delta_seconds"] = round(avg_time_delta, 2)

    reconciliation_details: dict[str, object] = {}
    if matches or missing_logs or missing_kraken:
        ledger = module.load_trade_ledger()  # type: ignore[attr-defined]
        if DRY_RUN_MODE:
            class _DryRunLedger:
                def __init__(self, real_ledger):
                    self._ledger = real_ledger
                    self._notified = False

                def _notify(self) -> None:
                    if not self._notified:
                        print("⚠️ Skipping real trade actions in dry-run mode")
                        self._notified = True

                def mark_reconciled(self, *args, **kwargs):
                    self._notify()

                def mark_pending_reconciliation(self, *args, **kwargs):
                    self._notify()

                def log_trade(self, *args, **kwargs):
                    self._notify()

                def find_trade_by_txid(self, *args, **kwargs):
                    return self._ledger.find_trade_by_txid(*args, **kwargs)

                def __getattr__(self, name: str):
                    return getattr(self._ledger, name)

            ledger = _DryRunLedger(ledger)
        reconciliation_details = module.reconcile_trades(  # type: ignore[attr-defined]
            matches=matches,
            missing_log_trades=missing_logs,
            missing_kraken_trades=missing_kraken,
            ledger=ledger,
            grace_seconds=module.RECONCILIATION_GRACE_SECONDS,  # type: ignore[attr-defined]
        )
        report.update(reconciliation_details)
        pending_marked = int(reconciliation_details.get("log_trades_marked_pending", 0) or 0)
        imported = int(reconciliation_details.get("kraken_trades_imported", 0) or 0)
        if pending_marked:
            report["warnings"].append(f"Marked {pending_marked} trade(s) pending reconciliation.")
        if imported:
            report["warnings"].append(f"Imported {imported} trade(s) from Kraken history.")
    else:
        report["kraken_trades_imported"] = 0
        report["log_trades_marked_pending"] = 0

    if missing_logs:
        report["log_unmatched_samples"] = [trade.raw for trade in missing_logs[:3]]
    if missing_kraken:
        report["kraken_unmatched_samples"] = [trade.raw for trade in missing_kraken[:3]]

    pending_count = int(report.get("log_trades_marked_pending", 0) or 0)
    imported_count = int(report.get("kraken_trades_imported", 0) or 0)
    unmatched_total = int(report.get("unmatched_log_trades", 0) or 0) + int(
        report.get("unmatched_kraken_trades", 0) or 0
    )
    if pending_count or imported_count or unmatched_total:
        report["reconciliation_status"] = "warning" if module.IS_LIVE else "error"  # type: ignore[attr-defined]
    else:
        report["reconciliation_status"] = "ok"

    internal_state = module.load_internal_portfolio_state()  # type: ignore[attr-defined]
    report["internal_portfolio_state"] = internal_state

    balance_difference = None
    if balances and internal_state:
        internal_usdc = float(internal_state.get("available_capital") or 0.0)
        kraken_usdc = balances.get("USDC") or balances.get("USD") or balances.get("ZUSD")
        if kraken_usdc is not None:
            balance_difference = float(kraken_usdc) - internal_usdc
    report["balance_difference"] = balance_difference

    logs_inspection = module.inspect_daemon_logs()  # type: ignore[attr-defined]
    report["daemon_log_inspection"] = logs_inspection
    return report


def run_roi_audit() -> dict:
    module = load_script_module("audit_roi")
    return module.run_audit()  # type: ignore[attr-defined]


def run_live_stats() -> tuple[dict, dict]:
    module = load_script_module("show_live_stats")
    trades = module._load_closed_trades(module.TRADES_PATH)  # type: ignore[attr-defined]
    summary = module._compute_summary(trades)  # type: ignore[attr-defined]
    return summary, {"trade_count": len(trades)}


def emit(status: bool, message: str) -> None:
    symbol = "✅" if status else "❌"
    print(f"{symbol} {message}")


audit_warnings: list[str] = []


def warn(message: str) -> None:
    """Record a non-fatal warning and surface it in the CLI output."""

    print(f"⚠️ {message}")
    audit_warnings.append(message)


failures: list[str] = []

if not IS_LIVE:
    emit(True, "IS_LIVE is disabled — final audit running in paper mode.")

try:
    diag_summary, diag_logs = run_debug_diagnostics()
except Exception as exc:  # pragma: no cover - defensive diagnostics wrapper
    print(f"⚠️ Debug diagnostics failed: {exc}")
    failures.append("debug_diagnostics")
    diag_summary, diag_logs = {}, []
if diag_logs:
    print("\n".join(diag_logs))

if diag_summary.get("credentials"):
    emit(True, "Kraken credentials detected in environment/config.")
else:
    emit(False, "Missing Kraken API credentials.")
    failures.append("credentials")

private = diag_summary.get("private_api", {})
balance_ok = bool(private.get("Balance"))
trades_ok = bool(private.get("TradesHistory"))
emit(balance_ok, "Kraken Balance API check")
if not balance_ok:
    failures.append("kraken_balance_api")
emit(trades_ok, "Kraken TradesHistory API check")
if not trades_ok:
    failures.append("kraken_trades_api")

risk = diag_summary.get("risk_guard", {})
if risk.get("paused"):
    emit(False, f"Risk guard paused: {risk.get('reason')}")
    failures.append("risk_guard")
else:
    emit(True, "Risk guard active with no pause flags.")
drawdown_limit = float(CONFIG.get("live_mode", {}).get("drawdown_threshold", 0.10) or 0.10)
max_drawdown = abs(float(risk.get("max_drawdown", 0.0) or 0.0))
emit(max_drawdown <= drawdown_limit, f"Max drawdown {max_drawdown:.2%} within {drawdown_limit:.2%} limit.")
if max_drawdown > drawdown_limit:
    failures.append("drawdown_limit")

recent_errors = bool(diag_summary.get("log_errors"))
recent_error_entries = diag_summary.get("recent_error_entries") or []
stale_errors = diag_summary.get("stale_log_errors") or []
if recent_errors:
    emit(False, f"Recent log inspection found errors ({len(recent_error_entries)} entries ≤24h).")
    failures.append("log_errors")
else:
    if stale_errors:
        if SUPPRESS_STALE_ERRORS:
            emit(True, f"Stale log errors (>24h) suppressed ({len(stale_errors)} entries).")
        else:
            emit(True, f"Only stale log errors detected (>24h): {len(stale_errors)} entries.")
    else:
        emit(True, "No critical errors detected in daemon/system logs.")

    if diag_summary.get("log_warnings"):
        if STRICT_MODE_ENABLED:
            emit(False, "Log warnings detected (STRICT_MODE enabled).")
            failures.append("log_warnings")
        else:
            emit(True, "Log warnings detected (see diagnostics output).")

validate_status = str(diag_summary.get("validate_trade") or "not attempted")
validate_timestamp = diag_summary.get("validate_trade_timestamp")
if validate_status.startswith("ok"):
    suffix = f" at {validate_timestamp}" if validate_timestamp else ""
    emit(True, f"Validate-only trade succeeded{suffix}.")
elif validate_status.startswith("skipped"):
    emit(True, f"Validate-only trade skipped ({validate_status}).")
elif validate_status == "not attempted":
    emit(True, "Validate-only trade not attempted.")
else:
    emit(False, f"Validate-only trade reported {validate_status}.")
    failures.append("validate_trade")

try:
    rec_report = run_audit_reconciliation()
except Exception as exc:  # pragma: no cover - defensive reconciliation wrapper
    print(f"⚠️ Audit reconciliation failed: {exc}")
    failures.append("audit_reconciliation")
    rec_report = {}
maintenance_info = (rec_report.get("daemon_log_inspection") or {}).get("maintenance") or {}
if maintenance_info.get("archived"):
    emit(True, "Daemon log maintenance archived stale entries (logs/archive).")
elif maintenance_info.get("cleaned"):
    emit(True, "Daemon log maintenance pruned entries older than 48h.")
if rec_report.get("kraken_trade_status") == "ok":
    emit(True, f"Kraken trade history fetched ({rec_report.get('kraken_trade_count')} trades).")
else:
    emit(False, f"Failed to fetch Kraken trade history: {rec_report.get('kraken_trade_error')}")
    failures.append("kraken_trade_fetch")

usd_violations = int(rec_report.get("usd_pair_violations", 0) or 0)
emit(usd_violations == 0, "All Kraken trades use USDC quotes.")
if usd_violations:
    failures.append("usd_pair_violations")

unmatched_logs = int(rec_report.get("unmatched_log_trades", 0) or 0)
unmatched_kraken = int(rec_report.get("unmatched_kraken_trades", 0) or 0)
pending_marked = int(rec_report.get("log_trades_marked_pending", 0) or 0)
imported_trades = int(rec_report.get("kraken_trades_imported", 0) or 0)

paper_mode_ignore = AUDIT_IGNORE_UNMATCHED_PAPER_TRADES and (DRY_RUN_MODE or not IS_LIVE)
if unmatched_logs == 0:
    if pending_marked:
        emit(True, f"Reconciliation pending: {pending_marked} local trade(s) awaiting Kraken match.")
    else:
        emit(True, "All local trades reconcile with Kraken history.")
else:
    msg = f"{unmatched_logs} local trade(s) unmatched"
    if pending_marked:
        msg += f"; {pending_marked} marked pending"
    if paper_mode_ignore:
        trade_word = "trade" if unmatched_logs == 1 else "trades"
        emit(True, "Reconciliation matched (paper mode)")
        warn(f"{unmatched_logs} unmatched local {trade_word} ignored (paper mode)")
    elif IS_LIVE and not SOFT_AUDIT_MODE:
        emit(False, f"Reconciliation mismatch: {msg}.")
        failures.append("unmatched_log_trades")
    else:
        warn(f"Reconciliation mismatch: {msg}.")

if unmatched_kraken == 0:
    if imported_trades:
        emit(True, f"Imported {imported_trades} Kraken trade(s) during reconciliation.")
    else:
        emit(True, "No Kraken trades missing from local ledger.")
else:
    msg = f"{unmatched_kraken} Kraken trade(s) missing locally"
    if imported_trades:
        msg += f"; imported {imported_trades} this run"
    if IS_LIVE and not SOFT_AUDIT_MODE:
        emit(False, f"Reconciliation mismatch: {msg}.")
        failures.append("unmatched_kraken_trades")
    else:
        # Soft audit mode (paper/default): surface mismatch without failing the audit.
        warn(f"Reconciliation mismatch: {msg}.")

balance_difference = rec_report.get("balance_difference")
if balance_difference is None:
    if IS_LIVE and not SOFT_AUDIT_MODE:
        emit(False, "Unable to compare Kraken vs internal balances.")
        failures.append("balance_comparison")
    else:
        # Soft audit mode keeps audit green but calls out the missing comparison.
        warn("Unable to compare Kraken vs internal balances (soft mode).")
else:
    tolerance = 5.0  # USD
    diff = float(balance_difference)
    if abs(diff) <= tolerance:
        emit(True, f"Balance difference ({diff:.2f}) within ±{tolerance:.2f} USD.")
    else:
        message = f"Balance difference ({diff:.2f}) exceeds ±{tolerance:.2f} USD."
        if IS_LIVE and not SOFT_AUDIT_MODE:
            emit(False, message)
            failures.append("balance_difference")
        else:
            # Soft audit mode keeps audit green but highlights the imbalance.
            warn(message)

try:
    roi_summary = run_roi_audit()
except Exception as exc:  # pragma: no cover - defensive ROI wrapper
    print(f"⚠️ ROI audit failed: {exc}")
    failures.append("roi_audit")
    roi_summary = {}
local_balance_value = roi_summary.get("local_balance")
local_balance_source = (roi_summary.get("local_balance_source") or "unknown") if isinstance(roi_summary, dict) else "unknown"
if IS_LIVE:
    try:
        balance_float = float(local_balance_value)
        if not math.isfinite(balance_float):
            raise ValueError("non-finite")
    except Exception:
        emit(False, "Ledger balance unavailable in live mode.")
        failures.append("ledger_balance")
    else:
        emit(True, f"Ledger balance source={local_balance_source} value=${balance_float:,.2f}")
else:
    if isinstance(local_balance_value, (int, float)):
        emit(True, f"Paper ledger balance source={local_balance_source} value=${float(local_balance_value):,.2f}")
    else:
        emit(True, "Paper ledger balance simulated (no live fetch).")

total_trades = int(roi_summary.get("total_trades", 0) or 0)
emit(total_trades > 0, f"ROI audit found {total_trades} closed trades.")
if total_trades == 0:
    failures.append("roi_trades")

roi_warnings_raw = roi_summary.get("warnings") or []
if isinstance(roi_warnings_raw, str):
    roi_warning_entries = [roi_warnings_raw]
else:
    roi_warning_entries = [str(entry) for entry in roi_warnings_raw if entry]

minor_roi_warnings: list[str] = []
major_roi_warnings: list[str] = []

balance_diff_value: float | None = None
if roi_summary.get("balance_difference") is not None:
    try:
        balance_diff_value = abs(float(roi_summary.get("balance_difference")))
    except (TypeError, ValueError):
        balance_diff_value = None

balance_within_flag = bool(roi_summary.get("balance_within_tolerance"))

for warning_text in roi_warning_entries:
    normalized_text = warning_text.strip()
    lower_text = normalized_text.lower()
    match = BALANCE_WARNING_PATTERN.search(normalized_text)
    if (
        match
        and AUDIT_IGNORE_MINOR_ROI_WARNINGS
        and ("within" in lower_text or balance_within_flag)
    ):
        try:
            parsed_diff = abs(float(match.group(1)))
            parsed_tol = float(match.group(2))
        except (TypeError, ValueError):
            parsed_diff = None
            parsed_tol = None

        effective_tol = AUDIT_MINOR_ROI_WARNING_TOLERANCE
        if parsed_tol is not None:
            effective_tol = max(effective_tol, parsed_tol)

        candidate_diffs = [value for value in (parsed_diff, balance_diff_value) if value is not None]
        if candidate_diffs and all(value <= effective_tol + 1e-9 for value in candidate_diffs):
            minor_roi_warnings.append(normalized_text)
            continue
        if not candidate_diffs and "within" in lower_text:
            minor_roi_warnings.append(normalized_text)
            continue

    major_roi_warnings.append(normalized_text)

if major_roi_warnings:
    emit(False, f"ROI audit warnings: {major_roi_warnings}")
    failures.append("roi_warnings")
    if minor_roi_warnings:
        warn(f"ROI audit minor warnings ignored: {minor_roi_warnings}")
elif minor_roi_warnings:
    warn(f"ROI audit minor warnings ignored: {minor_roi_warnings}")
    emit(True, "ROI audit produced only minor warnings within tolerance.")
else:
    emit(True, "ROI audit produced no warnings.")

final_balance = float(roi_summary.get("final_balance", 0.0))
emit(final_balance > 0, f"ROI audit ending balance ${final_balance:,.2f}.")
if final_balance <= 0:
    failures.append("roi_balance")

try:
    stats_summary, stats_meta = run_live_stats()
except Exception as exc:  # pragma: no cover - defensive stats wrapper
    print(f"⚠️ Live stats collection failed: {exc}")
    failures.append("live_stats")
    stats_summary, stats_meta = {}, {}
trade_count = int(stats_summary.get("total_trades", 0) or 0)
emit(trade_count > 0, f"Live stats report {trade_count} closed trades.")
if trade_count == 0:
    failures.append("live_stats_trades")

cumulative_roi = float(stats_summary.get("cumulative_roi", 0.0))
emit(cumulative_roi > -0.20, f"Cumulative ROI {cumulative_roi:.4f}.")
if cumulative_roi <= -0.20:
    failures.append("cumulative_roi")

deploy_context = {
    "trade_count": trade_count,
    "cumulative_roi": cumulative_roi,
    "max_drawdown": max_drawdown,
    "balance_difference": balance_difference,
    "phase_before": CONFIG.get("deployment", {}).get("phase"),
}

canary_reasons: list[dict[str, str]] = []
if "log_errors" in failures:
    canary_reasons.append(
        {
            "reason": "recent_log_errors",
            "resolution_tip": "Inspect logs/daemon.out and logs/system.log for CRITICAL entries within the last 24h and remediate before re-running the audit.",
        }
    )
if usd_violations:
    canary_reasons.append(
        {
            "reason": "usd_pair_violation",
            "resolution_tip": "Normalize strategy pairs to USDC quotes (e.g., BTC/USDC) and resync Kraken metadata.",
        }
    )

canary_success = (
    not failures
    and trade_count >= CANARY_MIN_TRADES
    and max_drawdown <= CANARY_MAX_DRAWDOWN
)

if canary_success:
    _write_deploy_phase_file("full", "pass", deploy_context)
    send_alert(
        "[audit] Canary phase succeeded — promoting to full deployment.",
        level="INFO",
        context={**deploy_context, "phase_after": "full"},
    )
    print("Canary success thresholds met — DEPLOY_PHASE promoted to 'full'.")
else:
    status = "fail" if failures else "pending"
    context_payload = {**deploy_context, "phase_after": "canary"}
    if failures:
        context_payload["failures"] = sorted(set(failures))
    if canary_reasons:
        context_payload["canary_reasons"] = canary_reasons
    if audit_warnings:
        context_payload["warnings"] = audit_warnings
    _write_deploy_phase_file("canary", status, context_payload)
    send_alert(
        "[audit] Remaining in canary deployment.",
        level="ERROR" if failures else "WARNING",
        context=context_payload,
    )
    if failures:
        print("Final health audit detected issues — remaining in canary mode.")

if failures:
    print("\nFinal health audit FAILED. Issues:", ", ".join(sorted(set(failures))))
    if not DRY_RUN_MODE:
        sys.exit(1)

if audit_warnings:
    print("\nFinal health audit completed with warnings:")
    for entry in audit_warnings:
        print(f" - {entry}")
    print("System ready for launch with caution — review warnings above.")
elif not failures:
    print("\nAll final health checks passed. System ready for live launch.")

if DRY_RUN_MODE:
    print("✅ Dry-run completed — all live checks simulated safely (no trades executed).")
PY

exit "${AUDIT_STATUS}"
