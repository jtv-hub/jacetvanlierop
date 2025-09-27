#!/usr/bin/env python3
"""Archive or clear ``logs/alerts.log`` when it grows beyond a threshold."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")
ALERT_LOG = LOG_DIR / "alerts.log"
ARCHIVE_DIR = LOG_DIR / "archive"
SYSTEM_LOG = LOG_DIR / "system.log"
DEFAULT_THRESHOLD = 5000
HEADER_TEMPLATE = "# alerts.log reset at {timestamp} UTC (threshold={threshold}, mode={mode})\n"


def _configure_logging() -> None:
    """Route script diagnostics into ``logs/system.log`` for operator review."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [clear_alerts] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(SYSTEM_LOG, encoding="utf-8")],
        force=True,
    )


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _write_header(path: Path, *, threshold: int, mode: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(HEADER_TEMPLATE.format(timestamp=timestamp, threshold=threshold, mode=mode), encoding="utf-8")


def _rotate_alerts(threshold: int, *, force: bool) -> tuple[bool, bool]:
    """Rotate alerts.log if threshold exceeded or force requested.

    Returns ``True`` when rotation occurred, ``False`` otherwise.
    """
    if not ALERT_LOG.exists():
        logging.info("alerts.log not found; nothing to archive.")
        return False, False

    try:
        line_count = _count_lines(ALERT_LOG)
    except OSError as exc:  # permission or IO issues
        logging.error("Unable to read alerts.log: %s", exc)
        return False, True

    if not force and line_count <= threshold:
        logging.info(
            "alerts.log within threshold (lines=%d threshold=%d); leaving in place.",
            line_count,
            threshold,
        )
        return False, False

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = ARCHIVE_DIR / f"alerts_{timestamp}.log"

    try:
        ALERT_LOG.replace(archive_path)
        _write_header(ALERT_LOG, threshold=threshold, mode="force" if force else "threshold")
        logging.info("alerts.log rotated to %s (lines=%d).", archive_path, line_count)
        return True, False
    except OSError as exc:
        logging.error("Failed to rotate alerts.log: %s", exc)
        return False, True


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive alerts.log when oversized.")
    parser.add_argument(
        "--archive-if-over",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Archive alerts.log if it has more than this many lines (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rotate alerts.log regardless of current size.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _configure_logging()

    threshold = max(0, args.archive_if_over)
    rotated, fatal = _rotate_alerts(threshold, force=args.force)
    if rotated:
        logging.info("alerts.log reset complete.")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
