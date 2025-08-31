#!/usr/bin/env python3
"""
Lightweight, crash‑tolerant nightly review.

What it does (no external deps):
- Finds the newest JSONL feed in data/live/*BTCUSD*.jsonl
- Counts lines and makes a tiny summary
- Appends one CSV row to logs/snapshots/metrics_snapshots.csv
- Appends a human‑readable line to logs/reports/auto_review.log
- Returns non‑zero when there's no file or too few lines (so launchd can flag it)
- (Optional) Pops a macOS notification for human visibility
"""

from __future__ import annotations

import os
import sys
import subprocess  # moved to top-level to satisfy pylint C0415
from datetime import datetime
from pathlib import Path
from typing import Optional


def safe_mkdir(path: Path) -> None:
    """Create directory (and parents) if missing; ignore errors."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return


def latest_live_file(root: Path) -> Optional[Path]:
    """Return most recent BTCUSD *.jsonl file under data/live, or None."""
    live_dir = root / "data" / "live"
    if not live_dir.is_dir():
        return None
    try:
        candidates = [p for p in live_dir.glob("*BTCUSD*.jsonl") if p.is_file()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    except OSError:
        return None


def count_lines(file_path: Path) -> int:
    """Count lines of a text file, safely."""
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh)
    except (OSError, UnicodeError):
        return 0


def append_metrics_snapshot(
    root: Path,
    when: datetime,
    latest_file: Optional[Path],
    lines: int,
    note: str,
) -> None:
    """Append a single snapshot row to logs/snapshots/metrics_snapshots.csv."""
    snapshots_dir = root / "logs" / "snapshots"
    safe_mkdir(snapshots_dir)
    csv_path = snapshots_dir / "metrics_snapshots.csv"

    header = ["timestamp", "latest_file", "lines", "note"]

    try:
        if not csv_path.exists():
            with csv_path.open("w", encoding="utf-8") as w:
                w.write(",".join(header) + "\n")
    except OSError:
        return

    row = [
        when.strftime("%Y-%m-%d %H:%M:%S"),
        latest_file.name if latest_file is not None else "",
        str(lines),
        note,
    ]
    try:
        sanitized = [str(val).replace(",", " ") for val in row]
        with csv_path.open("a", encoding="utf-8") as w:
            w.write(",".join(sanitized) + "\n")
    except OSError:
        return


def write_human_log(root: Path, text: str) -> None:
    """Append one human‑readable line to logs/reports/auto_review.log."""
    reports_dir = root / "logs" / "reports"
    safe_mkdir(reports_dir)
    log_path = reports_dir / "auto_review.log"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with log_path.open("a", encoding="utf-8") as w:
            w.write(f"[{stamp}] {text}\n")
    except OSError:
        return


def notify_mac(msg: str) -> None:
    """Best-effort macOS notification (ignored on non-macOS or if osascript missing)."""
    try:
        subprocess.run(  # nosec - local system alert only
            [
                "osascript",
                "-e",
                f'display notification "{msg}" with title "Crypto Bot"',
            ],
            check=False,
            capture_output=True,
        )
    except (FileNotFoundError, OSError):
        pass


def main() -> int:
    """Run the minimal review and never raise."""
    root = Path(__file__).resolve().parents[0]
    if (root / "data").is_dir() and (root / "logs").is_dir():
        repo_root = root
    else:
        repo_root = root.parent

    now = datetime.now()
    live = latest_live_file(repo_root)
    min_expected = int(os.environ.get("MIN_LINES", "60"))

    if live is None:
        msg = "auto_review: no live BTCUSD jsonl files found (data/live)."
        print(f"{now:%Y-%m-%d %H:%M:%S} - WARNING - {msg}")
        write_human_log(repo_root, f"⚠️ {msg}")
        append_metrics_snapshot(repo_root, now, None, 0, "no_live_file")
        notify_mac("No BTC/USD live file found. Check capture.")
        return 3

    n_lines = count_lines(live)

    if n_lines < min_expected:
        msg = f"auto_review: too few lines ({n_lines} < {min_expected}) in {live.name}"
        print(f"{now:%Y-%m-%d %H:%M:%S} - WARNING - {msg}")
        write_human_log(repo_root, f"⚠️ {msg}")
        append_metrics_snapshot(repo_root, now, live, n_lines, "too_few_lines")
        notify_mac(f"Too few lines in latest capture ({n_lines} < {min_expected}).")
        return 2

    msg = f"auto_review: latest={live.name} lines={n_lines}"
    print(f"{now:%Y-%m-%d %H:%M:%S} - INFO - {msg}")
    write_human_log(repo_root, msg)
    append_metrics_snapshot(repo_root, now, live, n_lines, "ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pylint: disable=broad-exception-caught
        stamp_err = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{stamp_err} - ERROR - auto_review failed but continuing: {err}",
            file=sys.stderr,
        )
        raise SystemExit(0) from err
