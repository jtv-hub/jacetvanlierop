#!/usr/bin/env python3
"""Live diagnostics helper for Kraken trading bot.

Run with ``PYTHONPATH=src python scripts/debug_live_diagnostics.py`` to gather
quick health signals without touching the running daemon. The checks cover:

- Environment credentials (presence only; values are masked).
- Private Kraken API reachability (Balance and TradesHistory).
- Recent daemon/system logs for live-trade submissions and errors.
- Optional validate-only trade submission (never executes a live order).
- Summary of the findings for at-a-glance status.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Tuple

from crypto_trading_bot import config as bot_config
from crypto_trading_bot.config import CONFIG, ConfigurationError, set_live_mode
from crypto_trading_bot.safety.risk_guard import (
    check_pause as risk_guard_check_pause,
)
from crypto_trading_bot.safety.risk_guard import (
    load_state as risk_guard_load_state,
)
from crypto_trading_bot.safety.risk_guard import (
    state_path as risk_guard_state_path,
)
from crypto_trading_bot.utils.kraken_client import (
    KrakenAPIError,
    KrakenAuthError,
    kraken_client,
    kraken_place_order,
)

# Tail at most this many lines from each log file.
_LOG_TAIL_LIMIT = 200
_TRADELN_MARKER = "Submitting live trade | pair="
_ERROR_KEYWORDS = ("error", "warning", "critical", "exception", "traceback")
_DEFAULT_VALIDATE_SIZE = 0.001


def _mask(value: str | None) -> str:
    if not value:
        return "<missing>"
    trimmed = value.strip()
    return f"{trimmed[:6]}***" if len(trimmed) >= 6 else "***"


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_json(label: str, payload: Dict[str, Any]) -> None:
    print(f"{label}: {json.dumps(payload, indent=2, default=str)}")


def _check_env(summary: Dict[str, Any]) -> None:
    _print_header("Environment Variables")
    key = os.getenv("KRAKEN_API_KEY")
    secret = os.getenv("KRAKEN_API_SECRET")
    key_status = "present" if key else "missing"
    secret_status = "present" if secret else "missing"
    print(f"KRAKEN_API_KEY: {_mask(key)} ({key_status})")
    print(f"KRAKEN_API_SECRET: {_mask(secret)} ({secret_status})")
    summary["credentials"] = key_status == "present" and secret_status == "present"


def _query_private(
    method: str,
    params: Dict[str, Any] | None = None,
) -> Tuple[bool, Dict[str, Any] | str]:
    try:
        response = kraken_client.query_private(
            method,
            params=params or {},
            raise_for_error=False,
            return_result=False,
        )
        ok = bool(response.get("ok"))
        return ok, response
    except (KrakenAPIError, KrakenAuthError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except (TimeoutError, OSError, ValueError, RuntimeError) as exc:
        return False, f"Unexpected error: {exc}"


def _private_api_checks(summary: Dict[str, Any]) -> None:
    _print_header("Kraken Private API")
    results: Dict[str, bool] = {}
    for method in ("Balance", "TradesHistory"):
        ok, payload = _query_private(method, {})
        results[method] = ok
        if isinstance(payload, dict):
            _print_json(f"{method} response", payload)
        else:
            print(f"{method} error: {payload}")
    summary["private_api"] = results


def _tail_lines(path: Path, limit: int) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            buffer = deque(maxlen=limit)
            for line in handle:
                buffer.append(line.rstrip("\n"))
    except FileNotFoundError:
        print(f"{path} not found.")
        return []
    except OSError as exc:
        print(f"Failed to read {path}: {exc}")
        return []
    return list(buffer)


def _scan_logs(summary: Dict[str, Any]) -> None:
    _print_header("Log Inspection")
    trade_lines: list[str] = []
    error_lines: list[str] = []
    log_files = [Path("logs/daemon.out"), Path("logs/system.log")]
    for log_path in log_files:
        print(f"-- {log_path} --")
        lines = _tail_lines(log_path, _LOG_TAIL_LIMIT)
        if not lines:
            continue
        local_trade_lines: list[str] = []
        local_error_lines: list[str] = []
        for line in lines:
            lower = line.lower()
            if _TRADELN_MARKER in line:
                entry = f"{log_path}: {line}"
                trade_lines.append(entry)
                local_trade_lines.append(entry)
            if any(keyword in lower for keyword in _ERROR_KEYWORDS):
                entry = f"{log_path}: {line}"
                error_lines.append(entry)
                local_error_lines.append(entry)
        if local_trade_lines:
            print("Live trade submissions detected:")
            for entry in local_trade_lines:
                print(f"  {entry}")
        else:
            print("No live trade submissions found in tail segment.")
        if local_error_lines:
            print("Errors or warnings detected:")
            for entry in local_error_lines:
                print(f"  {entry}")
        else:
            print("No errors or warnings detected in tail segment.")
    summary["trade_lines"] = bool(trade_lines)
    summary["log_errors"] = bool(error_lines)


def _check_risk_guard(summary: Dict[str, Any]) -> None:
    _print_header("Risk Guard")
    state = risk_guard_load_state()
    paused, reason = risk_guard_check_pause(state)
    path = risk_guard_state_path()
    print(f"State file: {path}")
    print(f"Paused: {paused}")
    if reason:
        print(f"Reason: {reason}")
    print(
        f"Consecutive failures: {state.get('consecutive_failures', 0)} | "
        f"Max drawdown: {state.get('max_drawdown', 0.0):.4f}"
    )
    summary["risk_guard"] = {
        "paused": paused,
        "reason": reason,
        "consecutive_failures": state.get("consecutive_failures", 0),
        "max_drawdown": state.get("max_drawdown", 0.0),
    }


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_validate_trade() -> Tuple[str, float, float | None, float | None]:
    pairs = CONFIG.get("tradable_pairs") or []
    pair = pairs[0] if pairs else "BTC/USDC"
    base, _, quote = pair.partition("/")
    size = _coerce_float(CONFIG.get("dry_run_order_size"), _DEFAULT_VALIDATE_SIZE)
    if size <= 0:
        size = _DEFAULT_VALIDATE_SIZE
    quote_upper = quote.upper() if quote else "USDC"
    price_defaults = {
        "BTC": 25000.0,
        "ETH": 1800.0,
        "SOL": 25.0,
        "LINK": 15.0,
        "XRP": 0.6,
    }
    price = price_defaults.get(base.upper(), 1.0)
    if quote_upper not in {"USD", "USDC", "USDT"}:
        price = 0.1
    kraken_cfg = CONFIG.get("kraken", {}) or {}
    min_cost_threshold = _coerce_float(kraken_cfg.get("min_cost_threshold"), 0.5)
    pair_thresholds = kraken_cfg.get("pair_cost_minimums", {}) or {}
    threshold_override = pair_thresholds.get(pair)
    if threshold_override is not None:
        min_cost_threshold = _coerce_float(threshold_override, min_cost_threshold)
    return pair, size, price, min_cost_threshold


def _validate_trade(summary: Dict[str, Any]) -> None:
    _print_header("Validate-Only Trade")
    if not summary.get("credentials"):
        print("Skipping validate-only trade: credentials missing.")
        summary["validate_trade"] = "skipped_missing_credentials"
        return

    pair, size, price, threshold = _pick_validate_trade()
    original_live_state = bool(bot_config.is_live)
    live_enabled = False
    try:
        if not original_live_state:
            set_live_mode(True)
            live_enabled = True
    except ConfigurationError as exc:
        print(f"Unable to enable live mode for validation: {exc}")
    try:
        trade_response = kraken_place_order(
            pair,
            "buy",
            size,
            price=price,
            validate=True,
            min_cost_threshold=threshold,
        )
        _print_json("Validate-only order response", trade_response)
        ok = bool(trade_response.get("ok"))
        failure_detail = trade_response.get("error") or trade_response.get("code") or "unknown"
        summary["validate_trade"] = "ok" if ok else failure_detail
        if ok:
            print("Validate-only trade would have succeeded (response ok=True).")
        else:
            print("Validate-only trade reported failure; inspect response above.")
    except (KrakenAPIError, KrakenAuthError) as exc:
        print(f"Validate-only trade raised error: {exc}")
        summary["validate_trade"] = f"error: {exc}"
    except (TimeoutError, OSError, ValueError, RuntimeError) as exc:
        print(f"Unexpected error during validate-only trade: {exc}")
        summary["validate_trade"] = f"unexpected_error: {exc}"
    finally:
        if live_enabled:
            set_live_mode(False)


def _print_summary(summary: Dict[str, Any]) -> None:
    _print_header("Summary")
    credentials = "present" if summary.get("credentials") else "missing"
    private_api = summary.get("private_api", {})
    balance_status = "ok" if private_api.get("Balance") else "failed"
    trades_status = "ok" if private_api.get("TradesHistory") else "failed"
    trade_lines = "found" if summary.get("trade_lines") else "not found"
    log_errors = "detected" if summary.get("log_errors") else "none detected"
    validate = summary.get("validate_trade") or "not attempted"
    risk = summary.get("risk_guard", {})
    risk_status = "paused" if risk.get("paused") else "clear"

    print(f"Credentials: {credentials}")
    print(f"Balance API: {balance_status}")
    print(f"TradesHistory API: {trades_status}")
    print(f"Trade submission logs: {trade_lines}")
    print(f"Log errors/warnings: {log_errors}")
    print(f"Validate-only trade: {validate}")
    print(f"Risk guard: {risk_status}")
    if risk.get("paused") and risk.get("reason"):
        print(f"Risk guard reason: {risk.get('reason')}")


def main() -> int:
    """Run diagnostics and return ``0`` so the script can exit cleanly."""
    summary: Dict[str, Any] = {}
    _check_env(summary)
    _private_api_checks(summary)
    _scan_logs(summary)
    _check_risk_guard(summary)
    _validate_trade(summary)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
