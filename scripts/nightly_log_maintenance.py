"""
Nightly Log Maintenance

Runs three maintenance steps and aggregates results:
  1) fix_trade_log_integrity.py
     (repairs logs/trades.log -> logs/trades_fixed.log, positions_fixed.jsonl)
  2) remove_closed_positions.py
     (cleans positions_fixed.jsonl -> positions_cleaned.jsonl)
  3) run_sync_and_anomaly_check.py
     --trades logs/trades_fixed.log --positions logs/positions_cleaned.jsonl

All step outputs are appended to: logs/nightly_reports/YYYY-MM-DD.log

Sample cron (daily at 01:10):
  10 1 * * * /usr/bin/env python \
    /path/to/repo/scripts/nightly_log_maintenance.py \
    >> /path/to/repo/logs/nightly_reports/cron.out 2>&1

Sample launchd (macOS) command test:
  python scripts/nightly_log_maintenance.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

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
        """No-op when colorama is unavailable (keeps API stable)."""
        return None


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "nightly_reports"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_path_for_today() -> Path:
    """Return the nightly report path for today's date (UTC localtime)."""
    today = _dt.datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"{today}.log"


def _run_step(args_list: list[str]) -> Tuple[int, str, str]:
    """Run a subprocess with blocking semantics; capture stdout/stderr.

    Uses check=True to ensure we block until completion and surface non-zero exit codes.
    """
    try:
        proc = subprocess.run(
            [sys.executable, *args_list],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.CalledProcessError as e:  # propagate outputs on failure
        return e.returncode, e.stdout or "", e.stderr or str(e)
    except OSError as e:  # pragma: no cover
        return 1, "", str(e)


def _write_section(log_file: Path, title: str, stdout: str, stderr: str) -> None:
    with log_file.open("a", encoding="utf-8") as f:
        ts = _dt.datetime.now().isoformat()
        f.write(f"\n===== {title} @ {ts} =====\n")
        if stdout:
            f.write(stdout)
            if not stdout.endswith("\n"):
                f.write("\n")
        if stderr:
            f.write("[stderr]\n")
            f.write(stderr)
            if not stderr.endswith("\n"):
                f.write("\n")


def main() -> None:
    """Run the nightly maintenance steps and write a dated report log."""
    parser = argparse.ArgumentParser(description="Nightly log maintenance runner")
    parser.add_argument(
        "--email",
        default=None,
        help="Optional email to send summary (not implemented)",
    )
    args = parser.parse_args()

    colorama_init(autoreset=True)

    log_file = _log_path_for_today()

    # Step 1: Fix trade log integrity
    rc1, out1, err1 = _run_step([str(ROOT / "scripts" / "fix_trade_log_integrity.py")])
    _write_section(log_file, "Step 1: fix_trade_log_integrity", out1, err1)
    fix_pass = rc1 == 0
    # Fallback flush guard and existence check for positions_fixed.jsonl
    time.sleep(0.2)
    positions_fixed = ROOT / "logs" / "positions_fixed.jsonl"
    # Poll briefly (up to ~2s) until file exists and non-empty
    for _ in range(10):
        if positions_fixed.exists() and positions_fixed.stat().st_size > 0:
            break
        time.sleep(0.2)

    # Step 2: Remove closed positions (use fixed outputs)
    rc2, out2, err2 = _run_step([str(ROOT / "scripts" / "remove_closed_positions.py")])
    _write_section(log_file, "Step 2: remove_closed_positions", out2, err2)
    removal_pass = rc2 == 0

    # Step 3: Sync + anomaly check using fixed outputs
    sync_cmd = [
        str(ROOT / "scripts" / "run_sync_and_anomaly_check.py"),
        "--trades",
        str(ROOT / "logs" / "trades_fixed.log"),
        "--positions",
        str(ROOT / "logs" / "positions_cleaned.jsonl"),
        "--json",
    ]
    _, out3, err3 = _run_step(sync_cmd)
    _write_section(log_file, "Step 3: run_sync_and_anomaly_check --json", out3, err3)

    sync_pass = False
    anomaly_pass = False
    err_counts: Dict[str, int] = {}
    anomaly_counts: Dict[str, int] = {}
    try:
        data = json.loads(out3.strip() or "{}")
        sync_pass = bool((data.get("sync") or {}).get("pass", False))
        anomaly_pass = bool((data.get("anomalies") or {}).get("pass", False))
        err_counts = (data.get("sync") or {}).get("errors_by_category", {}) or {}
        anomaly_counts = (data.get("anomalies") or {}).get("by_category", {}) or {}
    except json.JSONDecodeError:
        pass

    # Console summary
    fix_emoji = "✅" if fix_pass else "❌"
    remove_emoji = "✅" if removal_pass else "❌"
    sync_emoji = "✅" if sync_pass else "❌"
    anom_emoji = "✅" if anomaly_pass else "❌"

    print(f"{Style.BRIGHT}=== Nightly Log Maintenance Summary ==={Style.RESET_ALL}")
    print(f"Fix pass:     {Fore.GREEN if fix_pass else Fore.RED}{fix_emoji}{Style.RESET_ALL}")
    print(f"Removal pass: {Fore.GREEN if removal_pass else Fore.RED}{remove_emoji}{Style.RESET_ALL}")
    print(f"Sync pass:    {Fore.GREEN if sync_pass else Fore.RED}{sync_emoji}{Style.RESET_ALL}")
    print(f"Anomaly pass: {Fore.GREEN if anomaly_pass else Fore.RED}{anom_emoji}{Style.RESET_ALL}")

    # If any failed, print and log category counts
    if not sync_pass or not anomaly_pass:
        with log_file.open("a", encoding="utf-8") as f:
            f.write("\n[Summary categories]\n")
            if not sync_pass and err_counts:
                f.write("Sync errors by category:\n")
                for k, v in err_counts.items():
                    f.write(f"  - {k}: {v}\n")
            if not anomaly_pass and anomaly_counts:
                f.write("Anomaly counts by category:\n")
                for k, v in anomaly_counts.items():
                    f.write(f"  - {k}: {v}\n")
        if not sync_pass and err_counts:
            print("Sync errors by category:")
            for k, v in err_counts.items():
                print(f"  - {k}: {v}")
        if not anomaly_pass and anomaly_counts:
            print("Anomaly counts by category:")
            for k, v in anomaly_counts.items():
                print(f"  - {k}: {v}")

    # Email flag stub
    if args.email:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"\n[Email] Requested summary to {args.email} (not implemented)\n")
        print(f"{Fore.YELLOW}Note:{Style.RESET_ALL} " f"--email provided but SMTP delivery is not implemented.")


if __name__ == "__main__":
    main()
