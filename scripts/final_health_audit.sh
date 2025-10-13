#!/usr/bin/env bash
set -euo pipefail

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

STRICT_MODE=$(python - <<'PY'
import os
print(int(os.getenv('STRICT_MODE', '0').strip().lower() in {'1','true','yes','on'}))
PY
)

"${PYTHON_BIN}" - <<'PY' || AUDIT_STATUS=$?
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
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
        CONFIG = getattr(config_module, "CONFIG", {})
        IS_LIVE = bool(getattr(config_module, "IS_LIVE", False))
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
if CONFIG_SOURCE != "module":
    print(f"⚠️ Configuration fallback active (source={CONFIG_SOURCE}).")
    for reason in CONFIG_ERRORS:
        print(f"   fallback_reason={reason}")

try:
    alerts_module = importlib.import_module("crypto_trading_bot.bot.utils.alerts")
    send_alert = alerts_module.send_alert
except Exception:  # pragma: no cover - best-effort fallback
    def send_alert(message: str, *, level: str = "INFO", context: dict | None = None) -> None:
        print(f"[alert:{level}] {message} context={context}")


DEPLOY_PHASE_FILE = ROOT / "logs" / "deploy_phase.json"
CANARY_MIN_TRADES = int(os.getenv("CRYPTO_TRADING_BOT_CANARY_MIN_TRADES", "10") or "10")
CANARY_MAX_DRAWDOWN = float(os.getenv("CRYPTO_TRADING_BOT_CANARY_MAX_DRAWDOWN", "0.01") or "0.01")


def _write_deploy_phase_file(phase: str, status: str, context: dict[str, Any] | None = None) -> None:
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
    report: dict[str, object] = {}
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
    if missing_logs:
        report["log_unmatched_samples"] = [trade.raw for trade in missing_logs[:3]]
    if missing_kraken:
        report["kraken_unmatched_samples"] = [trade.raw for trade in missing_kraken[:3]]

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

if diag_summary.get("log_errors"):
    emit(False, "Recent log inspection found errors.")
    failures.append("log_errors")
else:
    if diag_summary.get("log_warnings") and STRICT_MODE != "1":
        emit(True, "Log warnings detected (see diagnostics output).")
    elif diag_summary.get("log_warnings"):
        emit(False, "Log warnings detected (STRICT_MODE enabled).")
        failures.append("log_warnings")
    else:
        emit(True, "No critical errors detected in daemon/system logs.")

try:
    rec_report = run_audit_reconciliation()
except Exception as exc:  # pragma: no cover - defensive reconciliation wrapper
    print(f"⚠️ Audit reconciliation failed: {exc}")
    failures.append("audit_reconciliation")
    rec_report = {}
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
emit(unmatched_logs == 0, "All local trades reconcile with Kraken history.")
if unmatched_logs:
    failures.append("unmatched_log_trades")
emit(unmatched_kraken == 0, "No Kraken trades missing from local ledger.")
if unmatched_kraken:
    failures.append("unmatched_kraken_trades")

balance_difference = rec_report.get("balance_difference")
if balance_difference is None:
    emit(False, "Unable to compare Kraken vs internal balances.")
    failures.append("balance_comparison")
else:
    tolerance = 5.0  # USD
    diff = float(balance_difference)
    emit(abs(diff) <= tolerance, f"Balance difference ({diff:.2f}) within ±{tolerance:.2f} USD.")
    if abs(diff) > tolerance:
        failures.append("balance_difference")

try:
    roi_summary = run_roi_audit()
except Exception as exc:  # pragma: no cover - defensive ROI wrapper
    print(f"⚠️ ROI audit failed: {exc}")
    failures.append("roi_audit")
    roi_summary = {}
total_trades = int(roi_summary.get("total_trades", 0) or 0)
emit(total_trades > 0, f"ROI audit found {total_trades} closed trades.")
if total_trades == 0:
    failures.append("roi_trades")

if roi_summary.get("warnings"):
    emit(False, f"ROI audit warnings: {roi_summary['warnings']}")
    failures.append("roi_warnings")
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
    sys.exit(1)

print("\nAll final health checks passed. System ready for live launch.")
PY

exit "${AUDIT_STATUS}"
