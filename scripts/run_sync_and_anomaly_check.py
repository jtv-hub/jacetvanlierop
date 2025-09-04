"""
Run Sync and Anomaly Check

Combines:
- Sync checks between logs/trades.log and logs/positions.jsonl
- Anomaly checks using confidence_audit

Usage:
  python scripts/run_sync_and_anomaly_check.py \
      [--trades logs/trades.log] \
      [--positions logs/positions.jsonl] \
      [--json]

Outputs a human-readable summary, or JSON with --json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Tuple

try:
    from colorama import Fore, Style  # type: ignore[import-not-found]
    from colorama import init as colorama_init  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional

    class _Dummy:
        RESET_ALL = ""

    class _Fore(_Dummy):
        RED = GREEN = YELLOW = CYAN = ""

    class _Style(_Dummy):
        BRIGHT = NORMAL = ""

    Fore, Style = _Fore(), _Style()  # type: ignore

    def colorama_init(*_args, **_kwargs):  # type: ignore
        """No-op initializer used when colorama is unavailable."""
        return None


# Project imports (top-level). If package isn't installed, fall back to adding src/
try:
    from crypto_trading_bot.learning.confidence_audit import (
        audit_trades,
        summarize_anomalies,
    )
    from crypto_trading_bot.scripts.sync_validator import SyncValidator
except ImportError:  # pragma: no cover
    repo_src = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, os.path.abspath(repo_src))
    from crypto_trading_bot.learning.confidence_audit import (
        audit_trades,
        summarize_anomalies,
    )
    from crypto_trading_bot.scripts.sync_validator import SyncValidator


def _is_number(v) -> bool:
    """Return True if `v` can be parsed as float."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def build_sync_summary(
    trades_file: str,
    positions_file: str,
    validator_cls,
) -> Tuple[bool, Dict[str, Any]]:
    """Run sync checks and return (pass, details)."""
    validator = validator_cls(trades_file=trades_file, positions_file=positions_file)
    trades = validator.load_json_lines(trades_file, "trades.log")
    positions = validator.load_json_lines(positions_file, "positions.jsonl")

    # Indexes and helpers (normalize trade_id for robust matching)
    def _norm_id(x):
        if isinstance(x, str):
            return x.strip()
        if x is None:
            return None
        try:
            return str(x).strip()
        except (TypeError, ValueError):
            return None

    trades_ids = [_norm_id(t.get("trade_id")) for t in trades]
    pos_ids = [_norm_id(p.get("trade_id")) for p in positions]
    pos_id_set = {p for p in pos_ids if p}

    # Duplicate and missing-id detection
    dup_trades = [tid for tid, c in Counter(trades_ids).items() if tid and c > 1]
    dup_positions = [tid for tid, c in Counter(pos_ids).items() if tid and c > 1]
    missing_id_trades = sum(1 for tid in trades_ids if not tid)
    missing_id_positions = sum(1 for tid in pos_ids if not tid)

    # Closed/open trade validations
    closed = [t for t in trades if (t.get("status") or "").lower() == "closed"]
    open_trades = [t for t in trades if (t.get("status") or "").lower() != "closed"]
    errors_by_cat: Dict[str, int] = {}
    debug_ids: Dict[str, list] = {
        "closed_trade_still_in_positions": [],
        "open_trade_missing_in_positions": [],
    }

    for t in closed:
        tid = _norm_id(t.get("trade_id"))
        # Closed trades should NOT remain in positions; flag if present
        if tid and tid in pos_id_set:
            errors_by_cat["closed_trade_still_in_positions"] = (
                errors_by_cat.get("closed_trade_still_in_positions", 0) + 1
            )
            if len(debug_ids["closed_trade_still_in_positions"]) < 10:
                debug_ids["closed_trade_still_in_positions"].append(tid)
        # exit_price present and numeric
        if not _is_number(t.get("exit_price")):
            errors_by_cat["closed_invalid_exit_price"] = errors_by_cat.get("closed_invalid_exit_price", 0) + 1
        # reason present and non-empty
        reason = t.get("reason")
        if not (isinstance(reason, str) and reason.strip()):
            errors_by_cat["closed_missing_exit_reason"] = errors_by_cat.get("closed_missing_exit_reason", 0) + 1
        # roi present and numeric
        if not _is_number(t.get("roi")):
            errors_by_cat["closed_invalid_roi"] = errors_by_cat.get("closed_invalid_roi", 0) + 1

    # Open trades should exist in positions
    for t in open_trades:
        tid = _norm_id(t.get("trade_id"))
        if not tid or tid not in pos_id_set:
            errors_by_cat["open_trade_missing_in_positions"] = (
                errors_by_cat.get("open_trade_missing_in_positions", 0) + 1
            )
            if len(debug_ids["open_trade_missing_in_positions"]) < 10 and tid:
                debug_ids["open_trade_missing_in_positions"].append(tid)

    # Malformed lines and other errors captured by validator
    malformed_count = sum(1 for e in validator.validation_errors if e.startswith("Malformed JSON"))

    sync_pass = not (
        errors_by_cat or dup_trades or dup_positions or missing_id_trades or missing_id_positions or malformed_count
    )

    details: Dict[str, Any] = {
        "duplicates": {
            "trades.log": dup_trades,
            "positions.jsonl": dup_positions,
        },
        "missing_trade_id": {
            "trades.log": missing_id_trades,
            "positions.jsonl": missing_id_positions,
        },
        "malformed_lines": malformed_count,
        "errors_by_category": errors_by_cat,
        "closed_trades_checked": len(closed),
        "debug_samples": debug_ids,
    }
    return sync_pass, details


def main():
    """CLI entry: run sync and anomaly checks and print summary."""
    parser = argparse.ArgumentParser(description="Run sync + anomaly checks on logs")
    parser.add_argument(
        "--trades",
        default="logs/trades_fixed.log",
        help="Path to trades JSONL (default: logs/trades_fixed.log)",
    )
    parser.add_argument(
        "--positions",
        default="logs/positions_cleaned.jsonl",
        help="Path to positions JSONL (default: logs/positions_cleaned.jsonl)",
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON summary")
    args = parser.parse_args()

    colorama_init(autoreset=True)

    # Sync summary
    sync_pass, sync_details = build_sync_summary(
        args.trades,
        args.positions,
        SyncValidator,
    )

    # Anomaly audit
    anomalies = audit_trades(args.trades, positions_file=args.positions)
    anomaly_summary = summarize_anomalies(
        anomalies,
        total_trades=sync_details.get("closed_trades_checked", 0),
    )
    anomalies_pass = anomaly_summary.get("anomalies_total", 0) == 0

    if args.json:
        out = {
            "sync": {"pass": sync_pass, **sync_details},
            "anomalies": {"pass": anomalies_pass, **anomaly_summary},
        }
        print(json.dumps(out, separators=(",", ":")))
        return

    # Human-readable output
    print(f"{Style.BRIGHT}=== Log Sync Check ==={Style.RESET_ALL}")
    status_label = "Sync Pass" if sync_pass else "Sync Fail"
    status_emoji = "✅" if sync_pass else "❌"
    print(f"{status_emoji} {status_label}")
    if not sync_pass:
        if sync_details["duplicates"]["trades.log"] or sync_details["duplicates"]["positions.jsonl"]:
            print(f"  {Fore.YELLOW}Duplicates:{Style.RESET_ALL}")
            if sync_details["duplicates"]["trades.log"]:
                print(f"    trades.log: {sync_details['duplicates']['trades.log']}")
            if sync_details["duplicates"]["positions.jsonl"]:
                print(f"    positions.jsonl: {sync_details['duplicates']['positions.jsonl']}")
        if any(sync_details["missing_trade_id"].values()):
            print(f"  {Fore.YELLOW}Missing trade_id entries:{Style.RESET_ALL} " f"{sync_details['missing_trade_id']}")
        if sync_details["malformed_lines"]:
            print(f"  {Fore.YELLOW}Malformed JSON lines:{Style.RESET_ALL} " f"{sync_details['malformed_lines']}")
        if sync_details["errors_by_category"]:
            print(f"  {Fore.RED}Errors by category:{Style.RESET_ALL}")
            for k, v in sync_details["errors_by_category"].items():
                print(f"    - {k}: {v}")
            # Print debug samples when available to visualize IDs
            samples = sync_details.get("debug_samples", {}) or {}
            for cat, ids in samples.items():
                if ids:
                    print(f"    > sample {cat}: {ids}")

    print(f"\n{Style.BRIGHT}=== Anomaly Detection ==={Style.RESET_ALL}")
    anomalies_label = "No anomalies" if anomalies_pass else "Anomalies Detected"
    anomalies_emoji = "✅" if anomalies_pass else "❌"
    print(f"{anomalies_emoji} {anomalies_label}")
    if not anomalies_pass:
        print("  Summary by category:")
        for k, v in sorted(
            anomaly_summary["by_category"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            print(f"    - {k}: {v}")


if __name__ == "__main__":
    main()
