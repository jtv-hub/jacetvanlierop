# ruff: noqa: E402
#!/usr/bin/env python3
"""
record_live_feed.py

Lightweight live market-data recorder.

- Uses data.adapters.kraken_public.KrakenPublicAdapter (public REST; no keys).
- Writes JSONL (default) or CSV snapshots to data/live/.
- Safe to Ctrl+C (SIGINT) out; it will close files cleanly.
- Includes basic retry/backoff and health checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, TextIO

# Make sure `data.adapters` is importable when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# We intentionally import after adjusting sys.path
from data.adapters.kraken_public import (
    KrakenPublicAdapter,
)  # pylint: disable=wrong-import-position

# ------------------------------- Utilities --------------------------------- #


def utc_now_str() -> str:
    """Return current UTC time formatted as YYYY-mm-dd HH:MM:SS."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def sanitize_pair(pair: str) -> str:
    """Return a filesystem-friendly version of a trading pair like BTC/USDC."""
    return pair.replace("/", "")


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------- Writers ----------------------------------- #


@dataclass
class JsonlWriter:
    """Append-only JSONL writer."""

    fh: TextIO

    def write(self, obj: Dict[str, Any]) -> None:
        """Write one JSON object per line."""
        self.fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.fh.flush()


@dataclass
class CsvWriter:
    """CSV writer that writes a header once and then rows."""

    fh: TextIO
    fieldnames: Iterable[str]
    _writer: Optional[csv.DictWriter] = None
    _has_header: bool = False

    def _ensure(self) -> None:
        """Initialize the csv writer and header if not already done."""
        if self._writer is None:
            self._writer = csv.DictWriter(self.fh, fieldnames=list(self.fieldnames))
        if not self._has_header:
            self._writer.writeheader()
            self._has_header = True

    def write(self, obj: Dict[str, Any]) -> None:
        """Write a CSV row."""
        self._ensure()
        assert self._writer is not None
        # Only include known columns; drop extras rather than failing.
        row = {k: obj.get(k, "") for k in self._writer.fieldnames}
        self._writer.writerow(row)
        self.fh.flush()


# ------------------------------- Collector --------------------------------- #


@dataclass
class CollectorConfig:
    """Configuration for the live data collector."""

    pair: str
    interval: float
    max_samples: Optional[int]
    out_dir: Path
    fmt: str
    print_every: int


class GracefulExit(Exception):
    """Raised internally to end the collection loop on SIGINT/SIGTERM."""


def _install_signal_handlers(raise_fn: Callable[[], None]) -> None:
    """Wire SIGINT and SIGTERM to raise a `GracefulExit` inside the loop."""

    def _handler(_signum, _frame):
        raise_fn()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def build_output_path(
    out_dir: Path,
    source: str,
    pair: str,
    fmt: str,
) -> Path:
    """
    Compute output file path including date, e.g.
    data/live/kraken_BTCUSD_20250812.jsonl.
    """
    ensure_dir(out_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = ".jsonl" if fmt == "jsonl" else ".csv"
    name = f"{source}_{sanitize_pair(pair)}_{stamp}{suffix}"
    return out_dir / name


def open_writer(path: Path, fmt: str) -> tuple[TextIO, Any]:
    """Open a writer (JSONL or CSV) and return (file_handle, writer_instance)."""
    fh = path.open("a", encoding="utf-8", newline="")
    if fmt == "jsonl":
        return fh, JsonlWriter(fh)
    # CSV fields we’ll write (keep order stable)
    fields = [
        "ts_client",
        "ts_exchange",
        "pair",
        "exchange_pair",
        "last",
        "bid",
        "ask",
        "volume_24h",
        "source",
    ]
    return fh, CsvWriter(fh, fields)


def collect_once(adapter: KrakenPublicAdapter, pair: str) -> Dict[str, Any]:
    """
    Fetch a single snapshot from the adapter and add a UTC string timestamp
    for human readability alongside numeric epoch fields.
    """
    payload = adapter.get_ticker(pair)
    # Add human-friendly timestamp
    payload["ts_utc"] = utc_now_str()
    return payload


def run_collector(cfg: CollectorConfig) -> None:
    """Main collection loop with basic backoff and periodic progress prints."""
    adapter = KrakenPublicAdapter()
    out_path = build_output_path(cfg.out_dir, "kraken", cfg.pair, cfg.fmt)
    fh, writer = open_writer(out_path, cfg.fmt)

    # Signals → raise inside loop
    def _raise_exit() -> None:
        raise GracefulExit()

    _install_signal_handlers(_raise_exit)

    # Health check
    if not adapter.ping():
        print("⚠️  Kraken public API did not respond to ping(). Proceeding anyway…")

    print(f"📡 Recording {cfg.pair} every {cfg.interval:.2f}s " f"→ {out_path} ({cfg.fmt.upper()})")

    backoff = 1.0
    written = 0
    started = time.time()

    try:
        while True:
            try:
                data = collect_once(adapter, cfg.pair)
                writer.write(data)
                written += 1
                backoff = 1.0  # reset on success

                if cfg.print_every and written % cfg.print_every == 0:
                    elapsed = time.time() - started
                    rate = written / max(elapsed, 1e-6)
                    last = data.get("last")
                    print(f"[{utc_now_str()}] samples={written} " f"rate={rate:.2f}/s last={last}")

                if cfg.max_samples is not None and written >= cfg.max_samples:
                    print(f"✅ Done: wrote {written} samples.")
                    return

            except GracefulExit:
                print(f"🛑 Stopped by signal. Wrote {written} samples.")
                return
            except (OSError, ValueError, RuntimeError) as exc:
                # Log and back off briefly to be polite to the API.
                print(f"⚠️  Error fetching/writing ({type(exc).__name__}): {exc}")
                time.sleep(min(backoff, 10.0))
                backoff *= 2.0

            # pacing
            time.sleep(max(cfg.interval, 0.0))

    finally:
        fh.close()


# --------------------------------- CLI ------------------------------------- #


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Record live public ticker data (Kraken).")
    p.add_argument(
        "--pair",
        default="BTC/USDC",
        help="Trading pair to record (default: %(default)s)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between requests (default: %(default)s)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N samples (default: run forever).",
    )
    p.add_argument(
        "--out-dir",
        default="data/live",
        help="Directory for output files (default: %(default)s)",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output format (default: %(default)s)",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print progress every N samples (0 to disable). " "Default: %(default)s",
    )
    return p.parse_args(argv)


def main() -> None:
    """Entrypoint: build config and run the collector."""
    args = parse_args()
    cfg = CollectorConfig(
        pair=args.pair,
        interval=args.interval,
        max_samples=args.max_samples,
        out_dir=Path(args.out_dir),
        fmt=args.fmt,
        print_every=int(args.print_every),
    )
    run_collector(cfg)


if __name__ == "__main__":
    main()
