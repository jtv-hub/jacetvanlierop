#!/usr/bin/env python3
"""
Compute position size for paper/live runs.

Reinvestment v1.1:
  size = clamp( MIN_SIZE,
                BASE_SIZE * (1 + ret_pct/100)^ALPHA,
                MAX_SIZE )
Where ret_pct comes from the last row of logs/paper/paper_daily.csv
(if missing, defaults to 0). This gives smooth compounding with caps.

Env vars (all optional):
  BASE_SIZE   : float, default 25
  MIN_SIZE    : float, default BASE_SIZE * 0.4
  MAX_SIZE    : float, default BASE_SIZE * 4.0
  ALPHA       : float, default 0.5   (0=no scaling, 1=full scaling)
  STEP        : float, default 1.0   (round down to nearest STEP)
  ECHO_REASON : "1" to print a short reason line to stdout
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
CSV_DAILY = REPO / "logs" / "paper" / "paper_daily.csv"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _load_last_ret_pct() -> Optional[float]:
    """Return last ret_pct from paper_daily.csv (column 15 in your pipeline)."""
    if not CSV_DAILY.exists():
        return None
    try:
        last_row = None
        with CSV_DAILY.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                last_row = row
        if not last_row:
            return None
        # Column 15 (0-based 14) was used earlier for ret_pct in your scripts.
        # Guard for short rows.
        if len(last_row) >= 15:
            return float(last_row[14])
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _round_down(v: float, step: float) -> float:
    if step <= 0:
        return v
    return math.floor(v / step) * step


def compute_size() -> Tuple[float, str]:
    """Compute reinvested size and a short reason string."""
    base = _env_float("BASE_SIZE", 25.0)
    min_sz = _env_float("MIN_SIZE", base * 0.4)
    max_sz = _env_float("MAX_SIZE", base * 4.0)
    alpha = _env_float("ALPHA", 0.5)
    step = _env_float("STEP", 1.0)

    ret_pct = _load_last_ret_pct()
    if ret_pct is None:
        # No paper history yet — use base with caps.
        size_raw = base
        reason = f"no-history → base={base:.2f}"
    else:
        # Equity multiple (1 + ret%) ^ alpha for smooth scaling.
        growth = max(0.0, 1.0 + (ret_pct / 100.0))
        mult = growth**alpha
        size_raw = base * mult
        reason = f"ret={ret_pct:.2f}% α={alpha:.2f} → mult={mult:.3f}"

    size = _round_down(_clamp(size_raw, min_sz, max_sz), step)
    return size, f"{reason} | clamp[{min_sz:.2f},{max_sz:.2f}] → {size:.2f}"


def main() -> None:
    """Entry point for computing and printing the position size.
    Prints the computed size to stdout, and optionally prints the
    calculation reason if the ECHO_REASON environment variable is set to "1".
    """
    size, reason = compute_size()
    print(f"{size:.2f}")
    if os.environ.get("ECHO_REASON") == "1":
        print(f"[size] {reason}")


if __name__ == "__main__":
    main()
