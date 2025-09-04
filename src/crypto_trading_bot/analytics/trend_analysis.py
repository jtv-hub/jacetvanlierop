"""Trend analytics for dashboard performance views.

Computes daily and weekly PnL, daily win rate, and a rolling 7-day ROI
average from a list of trade dicts (as parsed from JSONL).

Only CLOSED trades are considered. The logic is resilient to malformed rows
and missing fields.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple


def _to_dt(ts: str | None) -> datetime | None:
    """Parse ISO8601 string into aware datetime; returns None on failure."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


@dataclass
class _Agg:
    pnl: float = 0.0
    roi_sum: float = 0.0
    wins: int = 0
    count: int = 0


def _safe_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _iter_closed(trades: Iterable[dict]) -> Iterable[Tuple[datetime, float, float]]:
    for t in trades:
        try:
            if (t.get("status") or "").lower() != "closed":
                continue
            dt = _to_dt(t.get("timestamp"))
            if dt is None:
                continue
            size = _safe_float(t.get("size"))
            roi = _safe_float(t.get("roi"))
            if size is None or roi is None:
                continue
            yield dt, size, roi
        except Exception:  # pragma: no cover - defensive skip
            continue


def compute_trends(trades: List[dict]) -> Dict:
    """Compute daily/weekly PnL, daily win rate, and 7-day rolling ROI.

    Returns a dict with keys:
    - daily: [{date, pnl, roi, win_rate, trades}]
    - weekly: [{week, pnl, roi, win_rate, trades}]  # week in YYYY-Www
    - rolling_7d: [{date, roi}]
    """
    daily: Dict[str, _Agg] = defaultdict(_Agg)
    weekly: Dict[str, _Agg] = defaultdict(_Agg)

    # Aggregate
    for dt, size, roi in _iter_closed(trades):
        dkey = dt.date().isoformat()
        wkey = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
        pnl = size * roi
        for agg in (daily[dkey], weekly[wkey]):
            agg.pnl += pnl
            agg.roi_sum += roi
            agg.wins += 1 if roi > 0 else 0
            agg.count += 1

    # Materialize ordered daily list
    days_sorted = sorted(daily.keys())
    daily_list = []
    for d in days_sorted:
        a = daily[d]
        roi_avg = (a.roi_sum / a.count) if a.count else 0.0
        win_rate = (a.wins / a.count) if a.count else 0.0
        daily_list.append(
            {
                "date": d,
                "pnl": round(a.pnl, 6),
                "roi": round(roi_avg, 6),
                "win_rate": round(win_rate, 6),
                "trades": a.count,
            }
        )

    # Weekly list
    weeks_sorted = sorted(weekly.keys())
    weekly_list = []
    for w in weeks_sorted:
        a = weekly[w]
        roi_avg = (a.roi_sum / a.count) if a.count else 0.0
        win_rate = (a.wins / a.count) if a.count else 0.0
        weekly_list.append(
            {
                "week": w,
                "pnl": round(a.pnl, 6),
                "roi": round(roi_avg, 6),
                "win_rate": round(win_rate, 6),
                "trades": a.count,
            }
        )

    # Rolling 7-day ROI average over daily roi values
    rolling_7d = []
    roi_series = [x["roi"] for x in daily_list]
    for i, d in enumerate(days_sorted):
        start = max(0, i - 6)
        window = roi_series[start : i + 1]
        avg = sum(window) / len(window) if window else 0.0
        rolling_7d.append({"date": d, "roi": round(avg, 6)})

    return {"daily": daily_list, "weekly": weekly_list, "rolling_7d": rolling_7d}


__all__ = ["compute_trends"]
