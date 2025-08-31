#!/usr/bin/env python3
"""
cleanup_logs.py

Housekeeping for logs, reports, snapshots, and backups.

Retention policy (defaults):
- Keep reports (*.txt) for last 30 days
- Keep P&L snapshots (*.csv) for last 30 days
- Keep last 10 backup folders under logs/backups/
- Rotate trades.log and errors.log when > 10 MB, keep 5 versions
- Never delete: learning_ledger.jsonl, confidence_summary.json, paper_trades.jsonl

Usage:
  python scripts/cleanup_logs.py --dry-run
  python scripts/cleanup_logs.py --reports-days 45 --snapshots-days 60 --backups-keep 15
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = LOGS_DIR / "reports"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
BACKUPS_DIR = LOGS_DIR / "backups"

# Critical files that must never be removed
PROTECTED = {
    LOGS_DIR / "learning_ledger.jsonl",
    LOGS_DIR / "confidence_summary.json",
    LOGS_DIR / "paper_trades.jsonl",
}

# Rotatable log files (size-based)
ROTATE_LOGS = [
    LOGS_DIR / "trades.log",
    LOGS_DIR / "errors.log",
]


def human_bytes(n: int) -> str:
    """Return a short, human-readable byte string (e.g., 1.2MB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def find_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    """Collect files under a folder that match any of the glob patterns."""
    out: List[Path] = []
    if not root.exists():
        return out
    for pat in patterns:
        out.extend(sorted(root.glob(pat)))
    return out


def older_than(path: Path, cutoff: datetime) -> bool:
    """True if path's mtime is older than cutoff; False if missing."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
    except FileNotFoundError:
        return False
    return mtime < cutoff


def delete_paths(paths: Iterable[Path], dry_run: bool) -> None:
    """Delete a list of files/dirs, respecting the PROTECTED set."""
    for p in paths:
        if p in PROTECTED:
            logging.debug("Skipping protected file: %s", p)
            continue
        if not p.exists():
            continue
        if p.is_dir():
            logging.info("🗑️  Removing directory: %s", p)
            if not dry_run:
                shutil.rmtree(p, ignore_errors=True)
        else:
            logging.info("🗑️  Removing file: %s", p)
            if not dry_run:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass


def prune_by_age(
    folder: Path, patterns: Iterable[str], days: int, dry_run: bool
) -> Tuple[int, int]:
    """Remove files in `folder` older than `days`. Return (kept, removed)."""
    if not folder.exists():
        return 0, 0
    cutoff = datetime.now() - timedelta(days=days)
    candidates = find_files(folder, patterns)
    remove = [p for p in candidates if older_than(p, cutoff) and p not in PROTECTED]
    kept = len(candidates) - len(remove)
    if remove:
        logging.info(
            "Pruning by age in %s (>%d days): %d files", folder, days, len(remove)
        )
        delete_paths(remove, dry_run)
    return kept, len(remove)


def keep_last_n_dirs(folder: Path, keep: int, dry_run: bool) -> Tuple[int, int]:
    """Keep the most recent `keep` directories under `folder`. Return (kept, removed)."""
    if not folder.exists():
        return 0, 0
    dirs = [p for p in folder.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    to_keep = dirs[:keep]
    to_remove = dirs[keep:]
    if to_remove:
        logging.info(
            "Pruning backups in %s (keep last %d): removing %d dirs",
            folder,
            keep,
            len(to_remove),
        )
        delete_paths(to_remove, dry_run)
    return len(to_keep), len(to_remove)


def rotate_log_file(path: Path, max_bytes: int, versions: int, dry_run: bool) -> None:
    """
    Rotate `path` if larger than `max_bytes`:
    path.5 -> deleted, path.4 -> .5, ..., path -> .1 (and truncate current).
    """
    if not path.exists():
        return

    size = path.stat().st_size
    if size <= max_bytes:
        logging.debug(
            "No rotation needed for %s (%s <= %s)",
            path.name,
            human_bytes(size),
            human_bytes(max_bytes),
        )
        return

    logging.info(
        "🔁 Rotating %s (%s > %s)",
        path.name,
        human_bytes(size),
        human_bytes(max_bytes),
    )

    # Delete oldest
    oldest = path.with_suffix(path.suffix + f".{versions}")
    if oldest.exists():
        logging.info("🗑️  Removing oldest rotation: %s", oldest)
        if not dry_run:
            try:
                oldest.unlink()
            except FileNotFoundError:
                pass

    # Shift others
    for i in range(versions - 1, 0, -1):
        src = path.with_suffix(path.suffix + f".{i}")
        dst = path.with_suffix(path.suffix + f".{i+1}")
        if src.exists():
            logging.info("➡️  Renaming %s -> %s", src.name, dst.name)
            if not dry_run:
                try:
                    src.rename(dst)
                except FileNotFoundError:
                    pass

    # Move current to .1 and recreate empty current
    first = path.with_suffix(path.suffix + ".1")
    logging.info("➡️  Renaming %s -> %s", path.name, first.name)
    if not dry_run:
        try:
            path.rename(first)
        except FileNotFoundError:
            return
        path.touch()


def main() -> int:
    """CLI entry point for log retention & housekeeping."""
    parser = argparse.ArgumentParser(description="Log retention & housekeeping")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without deleting/rotating",
    )
    parser.add_argument(
        "--reports-days",
        type=int,
        default=30,
        help="Days to keep reports/*.txt (default: 30)",
    )
    parser.add_argument(
        "--snapshots-days",
        type=int,
        default=30,
        help="Days to keep snapshots/*.csv (default: 30)",
    )
    parser.add_argument(
        "--backups-keep",
        type=int,
        default=10,
        help="Number of latest backup dirs to keep (default: 10)",
    )
    parser.add_argument(
        "--max-log-size-mb",
        type=int,
        default=10,
        help="Rotate when log exceeds this size (default: 10MB)",
    )
    parser.add_argument(
        "--log-versions",
        type=int,
        default=5,
        help="How many rotated versions to keep (default: 5)",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    SNAPSHOTS_DIR.mkdir(exist_ok=True, parents=True)
    BACKUPS_DIR.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("🧹 Starting cleanup (dry_run=%s)", args.dry_run)

    # 1) Age-based pruning
    kept_r, removed_r = prune_by_age(
        REPORTS_DIR, ["*.txt"], args.reports_days, args.dry_run
    )
    kept_s, removed_s = prune_by_age(
        SNAPSHOTS_DIR, ["*.csv"], args.snapshots_days, args.dry_run
    )
    logging.info(
        "Reports: kept=%d, removed=%d | Snapshots: kept=%d, removed=%d",
        kept_r,
        removed_r,
        kept_s,
        removed_s,
    )

    # 2) Keep last N backups
    kept_b, removed_b = keep_last_n_dirs(BACKUPS_DIR, args.backups_keep, args.dry_run)
    logging.info("Backups: kept=%d, removed=%d", kept_b, removed_b)

    # 3) Size-based rotation for raw logs
    max_bytes = args.max_log_size_mb * 1024 * 1024
    for lf in ROTATE_LOGS:
        try:
            rotate_log_file(
                lf,
                max_bytes=max_bytes,
                versions=args.log_versions,
                dry_run=args.dry_run,
            )
        except (OSError, FileNotFoundError) as err:
            logging.warning("Could not rotate %s: %s", lf, err)

    logging.info("✅ Cleanup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
