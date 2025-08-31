#!/usr/bin/env python3
"""
Brute-force grid search over backtest parameters using the CLI backtester.

- Runs backtest.py many times on the same input file with different params
- Collects the appended rows from logs/backtests/backtest_summary.csv
- Writes a tidy results CSV: logs/backtests/grid_search_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from typing import List, Dict, Any


def _read_csv_rows(path: str) -> List[List[str]]:
    """Return all rows from a CSV file or an empty list if the file is missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def _latest_live_file(live_dir: str) -> str:
    """Find the newest JSONL file in the given live data directory."""
    if not os.path.isdir(live_dir):
        raise FileNotFoundError(f"Live dir not found: {live_dir}")
    files = [f for f in os.listdir(live_dir) if f.endswith(".jsonl")]
    if not files:
        raise FileNotFoundError(f"No JSONL files in {live_dir}")
    files.sort()
    return os.path.join(live_dir, files[-1])


def run_grid(
    infile: str,
    rsi_periods: List[int],
    rsi_ths: List[float],
    tps: List[float],
    sls: List[float],
    max_holds: List[int],
    size: float,
) -> List[Dict[str, Any]]:
    """
    Run the backtest over a small parameter grid and return collected rows
    from logs/backtests/backtest_summary.csv as dicts.
    """
    out_dir = os.path.join("logs", "backtests")
    os.makedirs(out_dir, exist_ok=True)

    summary_csv = os.path.join(out_dir, "backtest_summary.csv")
    before_rows = _read_csv_rows(summary_csv)
    header: List[str] = []
    if before_rows:
        header = before_rows[0]

    results: List[Dict[str, Any]] = []
    total = len(rsi_periods) * len(rsi_ths) * len(tps) * len(sls) * len(max_holds)
    i = 0

    for rp in rsi_periods:
        for th in rsi_ths:
            for tp in tps:
                for sl in sls:
                    for mh in max_holds:
                        i += 1
                        print(
                            f"[{i}/{total}] rp={rp} th={th} tp={tp} sl={sl} mh={mh}",
                            flush=True,
                        )
                        # Call the CLI backtester
                        cmd = [
                            sys.executable,
                            os.path.join(os.getcwd(), "backtest.py"),
                            "--file",
                            infile,
                            "--rsi-period",
                            str(rp),
                            "--rsi-th",
                            str(th),
                            "--tp",
                            str(tp),
                            "--sl",
                            str(sl),
                            "--max-hold",
                            str(mh),
                            "--size",
                            str(size),
                        ]
                        subprocess.run(cmd, check=True, capture_output=True, text=True)

                        # Read the tail row just appended
                        rows_after = _read_csv_rows(summary_csv)
                        if not rows_after:
                            continue
                        if not header:
                            header = rows_after[0]
                        # Grab the last data row
                        last = rows_after[-1]
                        if last and (last != before_rows[-1] if before_rows else True):
                            rec = dict(zip(header, last))
                            # attach a convenience score (wins * win_rate * expectancy)
                            try:
                                wr = float(rec.get("win_rate", "0"))
                                exp_pct = float(rec.get("expectancy_pct", "0"))
                                wins = float(rec.get("wins", "0"))
                                rec["score"] = f"{wins * wr * exp_pct:.6f}"
                            except (ValueError, TypeError, KeyError):
                                # Narrow exception handling to value/typing issues only
                                rec["score"] = ""
                            results.append(rec)
                        # keep our place up to date
                        before_rows = rows_after
                        # small pause so timestamp differences are clear in CSV
                        time.sleep(0.05)

    # Write a tidy copy of just this run's results
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"grid_search_results_{stamp}.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(r)
    print(f"✅ Grid done. {len(results)} rows → {out_csv}")
    return results


def main() -> int:
    """CLI entrypoint: parse args, locate the input file, and run the grid search."""
    ap = argparse.ArgumentParser(description="Grid search wrapper for backtest.py")
    ap.add_argument("--file", help="Input JSONL (defaults to latest in data/live)")
    ap.add_argument("--size", type=float, default=100.0)
    # sensible tiny grids to start (fast)
    ap.add_argument("--rsi-periods", default="10,14,21")
    ap.add_argument("--rsi-ths", default="55,60,65")
    ap.add_argument("--tps", default="0.0015,0.0020,0.0030")
    ap.add_argument("--sls", default="0.0025,0.0030,0.0040")
    ap.add_argument("--max-holds", default="8,10,14")
    args = ap.parse_args()

    infile = args.file or _latest_live_file(os.path.join("data", "live"))

    def _floats(s: str) -> List[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    def _ints(s: str) -> List[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    run_grid(
        infile=infile,
        rsi_periods=_ints(args.rsi_periods),
        rsi_ths=_floats(args.rsi_ths),
        tps=_floats(args.tps),
        sls=_floats(args.sls),
        max_holds=_ints(args.max_holds),
        size=args.size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
