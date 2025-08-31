#!/usr/bin/env python3
"""
Build an equity curve image from paper-trade logs.

- Reads:  logs/paper/paper_trades_*.jsonl   (one JSON object per line)
- Writes: logs/reports/equity_latest.png    (and a timestamped copy)

We treat each trade's `pnl_pct` as a return. If values look like percents
(e.g., 20.0 == 20%), we convert them to fractions (0.20). We also clip extreme
outliers to keep the curve readable.
"""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Dict, Any
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


REPO_ROOT = Path(".").resolve()
LOGS_DIR = REPO_ROOT / "logs"
LOG_PAPER = LOGS_DIR / "paper"
LOG_REPORTS = LOGS_DIR / "reports"


# ----------------------------- data loading ----------------------------------


def _iter_paper_jsonl(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    """Yield trade dicts from a sequence of jsonl files (skip malformed rows)."""
    for p in paths:
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Ignore broken lines; keep going.
                        continue
        except OSError:
            # If a file disappears between glob and open, just skip it.
            continue


def _load_closed_trades() -> List[Dict[str, Any]]:
    """Load all paper trades available under logs/paper/ (sorted by name)."""
    paths = sorted(LOG_PAPER.glob("paper_trades_*.jsonl"))
    return list(_iter_paper_jsonl(paths))


# --------------------------- equity computation ------------------------------


def _normalize_r(v: float) -> float:
    """
    Convert a raw value to a fractional return with light sanity checks.

    Accepts both fraction (0.20) and percent (20.0) style inputs. Clips
    to [-0.5, 0.5] so a single bad value can't destroy the chart scale.
    """
    try:
        r = float(v)
    except (TypeError, ValueError):
        return 0.0

    # Values between 1 and 100 likely represent percent, not fraction.
    if 1.0 < abs(r) <= 100.0:
        r = r / 100.0

    # Clip extreme outliers
    if r > 0.5:
        r = 0.5
    elif r < -0.5:
        r = -0.5
    return r


def _cum_equity(closed: List[Dict[str, Any]]) -> List[float]:
    """Compute cumulative equity starting at 1.0 from trade list."""
    equity: List[float] = [1.0]
    for tr in closed:
        r = _normalize_r(tr.get("pnl_pct", 0.0))
        equity.append(equity[-1] * (1.0 + r))
    return equity


# ------------------------------- rendering -----------------------------------


def _plot_equity(equity: List[float], out_path: Path) -> None:
    """Render the equity series to `out_path`, padding flat lines for visibility."""
    x = list(range(len(equity)))

    ymin = min(equity) if equity else 1.0
    ymax = max(equity) if equity else 1.0
    if ymin == ymax:
        # Pad a perfectly flat series so the line is visible.
        pad = 0.01 if ymin == 0 else 0.005
        ymin, ymax = ymin * (1 - pad), ymax * (1 + pad)
    else:
        # Add a small headroom/footroom around min/max.
        span = ymax - ymin
        ymin -= 0.03 * span
        ymax += 0.03 * span
        ymin = max(ymin, 0.0)  # keep baseline at/above 0 for readability

    plt.figure(figsize=(9, 4.5))
    if len(equity) >= 2:
        plt.plot(x, equity, linewidth=2)
    else:
        plt.plot(x, equity, linewidth=2, marker="o")

    plt.ylim(ymin, ymax)
    plt.title("Paper Trading — Equity (cumulative)")
    plt.xlabel("Closed trades")
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# ---------------------------------- main -------------------------------------


def main(*, verbose: bool = True) -> None:
    """Build the equity image from paper logs and save it in reports/."""
    LOG_REPORTS.mkdir(parents=True, exist_ok=True)

    closed = _load_closed_trades()
    equity = _cum_equity(closed)

    out_latest = LOG_REPORTS / "equity_latest.png"
    if verbose:
        mn = f"{min(equity):.4f}" if equity else "n/a"
        mx = f"{max(equity):.4f}" if equity else "n/a"
        print(
            "[equity] Loaded paper trades="
            f"{len(closed)}; plotting {len(equity)} equity points; "
            f"min={mn} max={mx}"
        )

    _plot_equity(equity, out_latest)

    # Also save a timestamped copy for history (optional but handy).
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_ts = LOG_REPORTS / f"equity_{ts}.png"
    try:
        out_ts.write_bytes(out_latest.read_bytes())
    except OSError:
        # If copy fails, continue; latest image is already written.
        pass

    if verbose:
        print(f"[equity] wrote {out_latest}")


if __name__ == "__main__":
    main(verbose=True)
