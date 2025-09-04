"""
Streamlit Dashboard for the Crypto Trading Bot

Mobile-friendly dashboard that summarizes live trading metrics, trends,
strategy leaderboards, and learning/shadow-test insights.

Run:
  streamlit run web/dashboard.py
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

try:
    import streamlit as st  # type: ignore[import-not-found]  # pylint: disable=import-error
except ImportError:  # pragma: no cover - dev environments without streamlit

    class _Stub:
        def __getattr__(self, name):  # noqa: D401 - tiny shim
            def _noop(*_args, **_kwargs):
                print(f"[streamlit-missing] called: {name}")

            return _noop

    st = _Stub()  # type: ignore[assignment]

# Optional pandas for nicer chart indexing; fall back if missing
try:
    import pandas as pd  # type: ignore[import-not-found]  # pylint: disable=import-error
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

# Optional numpy for explicit NaN handling in tables
try:
    import numpy as np  # type: ignore[import-not-found]  # pylint: disable=import-error
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


# ------------------------------
# Config and utilities
# ------------------------------

TRADES_LOG = os.path.join("logs", "trades.log")
LEARN_FEEDBACK = os.path.join("logs", "learning_feedback.jsonl")
SHADOW_RESULTS = os.path.join("logs", "shadow_test_results.jsonl")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def _fmt_pct(v: float | None) -> str:
    return f"{(v or 0.0) * 100:,.2f}%"


def _fmt_num(v: float | int | None) -> str:
    if v is None:
        return "0"
    try:
        return f"{v:,.0f}"
    except (TypeError, ValueError):
        return str(v)


# ------------------------------
# Trading metrics from trades.log
# ------------------------------


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # Handle both "...+00:00" and naive ISO formats
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError, AttributeError):
        return None


def _load_trades(path: str) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        roi = r.get("roi")
        if _is_num(roi):
            r["roi"] = float(roi)
        out.append(r)
    return out


def trade_metrics(trade_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute headline metrics, trends and leaderboard from trade rows."""
    closed = [t for t in trade_rows if (t.get("status") or "").lower() == "closed"]
    total = len(trade_rows)
    rois = [t.get("roi") for t in closed if _is_num(t.get("roi"))]
    rois_f = [float(x) for x in rois]
    wins = sum(1 for r in rois_f if r > 0)
    win_rate = (wins / len(rois_f)) if rois_f else 0.0
    avg_roi = (sum(rois_f) / len(rois_f)) if rois_f else 0.0
    cum = 1.0
    for r in rois_f:
        cum *= 1 + r
    cumulative_roi = (cum - 1.0) if rois_f else 0.0

    # Top strategies by count and ROI
    by_strat: Dict[str, List[float]] = defaultdict(list)
    for t in closed:
        s = t.get("strategy") or "Unknown"
        r = t.get("roi")
        if _is_num(r):
            by_strat[s].append(float(r))
    leaderboard: List[Tuple[str, int, float]] = []
    for s, vals in by_strat.items():
        cnt = len(vals)
        mean = sum(vals) / cnt if cnt else 0.0
        leaderboard.append((s, cnt, mean))
    leaderboard.sort(key=lambda x: (-x[1], -x[2]))

    # Last 24h activity
    now = datetime.now(timezone.utc)
    recent_any = False
    for t in trade_rows[-50:]:  # sample last 50 for speed
        dt = _parse_timestamp(t.get("timestamp"))
        if dt and (now - dt.replace(tzinfo=timezone.utc)) <= timedelta(hours=24):
            recent_any = True
            break

    # Daily/weekly trend series from trades (fallback if trend_analysis missing)
    daily_map: Dict[str, float] = defaultdict(float)
    for t in closed:
        dt = _parse_timestamp(t.get("timestamp"))
        if not dt or not _is_num(t.get("roi")):
            continue
        day = dt.date().isoformat()
        daily_map[day] += float(t["roi"])  # sum ROI per day
    days_sorted = sorted(daily_map.keys())
    daily_series = [(d, daily_map[d]) for d in days_sorted]

    # Weekly rollup (ISO week)
    weekly_map: Dict[str, float] = defaultdict(float)
    for d, v in daily_series:
        y, m, day = map(int, d.split("-"))
        iso = datetime(y, m, day, tzinfo=timezone.utc).isocalendar()
        key = f"{iso.year}-W{iso.week:02d}"
        weekly_map[key] += v
    weeks_sorted = sorted(weekly_map.keys())
    weekly_series = [(w, weekly_map[w]) for w in weeks_sorted]

    return {
        "total": total,
        "win_rate": win_rate,
        "avg_roi": avg_roi,
        "cumulative_roi": cumulative_roi,
        "leaderboard": leaderboard[:10],
        "daily_series": daily_series,
        "weekly_series": weekly_series,
        "recent_24h": recent_any,
    }


# ------------------------------
# Learning + shadow summaries
# ------------------------------


def summarize_learning(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize confidence change suggestions and top strategies."""
    inc = dec = same = 0
    deltas: List[float] = []
    strategies: List[str] = []
    for r in entries:
        cur = r.get("current_confidence") or r.get("confidence")
        sug = r.get("suggested_confidence") or r.get("suggested_value") or r.get("value")
        try:
            cur_f = float(cur) if cur is not None else None
            sug_f = float(sug) if sug is not None else None
        except (TypeError, ValueError):
            cur_f = None
            sug_f = None
        if cur_f is not None and sug_f is not None:
            d = sug_f - cur_f
            deltas.append(d)
            if d > 0:
                inc += 1
            elif d < 0:
                dec += 1
            else:
                same += 1
        strat = r.get("strategy") or r.get("strategy_name") or "Unknown"
        if isinstance(strat, str):
            strategies.append(strat)
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    top3 = [s for s, _ in Counter(strategies).most_common(3)]
    return {
        "total": len(entries),
        "inc": inc,
        "dec": dec,
        "same": same,
        "avg_delta": avg_delta,
        "top3": top3,
    }


def summarize_shadow(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize shadow tests (recent, best-in-10, >70% count, alerts)."""
    items: List[Tuple[float | None, int | None]] = []
    alert_low = False
    for r in entries:
        wr = r.get("win_rate") or r.get("success_rate")
        ne = r.get("num_exits") or r.get("trades_tested") or r.get("sample_size")
        try:
            wr_f = float(wr) if wr is not None else None
        except (TypeError, ValueError):
            wr_f = None
        try:
            ne_i = int(ne) if ne is not None else None
        except (TypeError, ValueError):
            ne_i = None
        if isinstance(wr_f, float) and wr_f < 0.5:
            alert_low = True
        items.append((wr_f, ne_i))
    recent5 = items[-5:]
    best10 = max((wr for wr, _ in items[-10:] if isinstance(wr, float)), default=None)
    gt_70 = sum(1 for wr, _ in items if isinstance(wr, float) and wr > 0.70)
    return {"recent5": recent5, "best10": best10, "gt70": gt_70, "alert": alert_low}


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

# Sidebar: refresh + file sizes
with st.sidebar:
    st.header("Status")
    now_txt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.write(f"Refreshed: {now_txt}")

    def _size(path: str) -> str:
        try:
            return f"{os.path.getsize(path):,} bytes"
        except OSError:
            return "missing"

    st.write(f"trades.log: {_size(TRADES_LOG)}")
    st.write(f"learning_feedback.jsonl: {_size(LEARN_FEEDBACK)}")
    st.write(f"shadow_test_results.jsonl: {_size(SHADOW_RESULTS)}")

# Load data
trades_data = _load_trades(TRADES_LOG)
learn_entries = _read_jsonl(LEARN_FEEDBACK)
shadow_entries = _read_jsonl(SHADOW_RESULTS)

tm = trade_metrics(trades_data)
ls = summarize_learning(learn_entries)
ss = summarize_shadow(shadow_entries)

# Alerts
if not tm.get("recent_24h"):
    st.warning("No trades detected in the last 24 hours.")
if ss.get("alert"):
    st.error("One or more shadow tests show win_rate below 50%.")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", _fmt_num(tm.get("total")))
col2.metric("Win Rate", _fmt_pct(tm.get("win_rate")))
col3.metric("Cumulative ROI", _fmt_pct(tm.get("cumulative_roi")))
col4.metric("Avg ROI / Trade", _fmt_pct(tm.get("avg_roi")))

# Trend charts
with st.expander("Trend (Daily/Weekly)", expanded=True):
    mode = st.radio("View", options=["Daily", "Weekly"], horizontal=True)
    series = tm.get("daily_series") if mode == "Daily" else tm.get("weekly_series")
    if series:
        labels = [k for k, _ in series]
        values = [v for _, v in series]

        # Convert weekly labels like 'YYYY-Www' to ISO week start dates for indexing
        if mode == "Weekly":

            def _week_to_date(label: str) -> str:
                try:
                    year_s, week_s = label.split("-W")
                    d = datetime.fromisocalendar(int(year_s), int(week_s), 1)
                    return d.date().isoformat()
                except Exception:  # pylint: disable=broad-exception-caught
                    return label

            labels = [_week_to_date(label) for label in labels]

        if pd is not None:
            df = pd.DataFrame({"ROI": values}, index=pd.to_datetime(labels))
            # Ensure Streamlit can serialize types cleanly
            df = df.convert_dtypes()
            st.line_chart(df)
        else:  # graceful fallback if pandas is not installed
            st.line_chart(values)
    else:
        st.info("No trend data available yet.")

# Leaderboard
with st.expander("Top Strategies", expanded=True):
    lb = tm.get("leaderboard") or []
    if lb:
        table_rows = [
            {
                "Strategy": name,
                "Trades": cnt,
                "Avg ROI": _fmt_pct(avg),
            }
            for name, cnt, avg in lb
        ]
        st.table(table_rows)
    else:
        st.info("No closed trades yet.")

# Learning + Shadow summaries
with st.expander("Learning Machine Summary", expanded=False):
    st.subheader("Learning Feedback")
    st.write(
        f"Suggestions: {_fmt_num(ls['total'])} | +{ls['inc']} / -{ls['dec']} / ={ls['same']} | "
        f"Avg Δ: {ls['avg_delta']:.3f}"
    )
    if ls.get("top3"):
        st.write("Top strategies: " + ", ".join(ls["top3"]))

    st.subheader("Shadow Tests")
    recent5_list = ss.get("recent5") or []
    if recent5_list:
        # Use None/NaN for missing numeric fields to avoid ArrowInvalid
        recent_table_rows = [
            {
                "win_rate": (wr if isinstance(wr, float) else None),
                "num_exits": (ne if isinstance(ne, int) else None),
            }
            for wr, ne in recent5_list
        ]
        if pd is not None:
            df_recent = pd.DataFrame(recent_table_rows)
            # Coerce numeric columns and normalize missing values
            if "num_exits" in df_recent.columns:
                df_recent["num_exits"] = pd.to_numeric(df_recent["num_exits"], errors="coerce")
            if "win_rate" in df_recent.columns:
                df_recent["win_rate"] = pd.to_numeric(df_recent["win_rate"], errors="coerce")
            if np is not None:
                df_recent = df_recent.fillna(value=np.nan)
            df_recent = df_recent.convert_dtypes()
            st.dataframe(df_recent)
        else:
            st.table(recent_table_rows)
    best10_val = ss.get("best10")
    st.write("Best win rate (last 10): " + (f"{best10_val:.2f}" if isinstance(best10_val, float) else "n/a"))
    st.write(f">70% win rate count: {ss.get('gt70', 0)}")

st.caption("Data updates as logs grow. Designed for mobile with columns and expanders.")
