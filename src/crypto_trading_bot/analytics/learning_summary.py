from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
    except OSError:
        return []
    return out


def _read_jsonl_tail(path: str, limit: int) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    return rows[-limit:] if limit and rows else rows


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _iso_date(ts: Any) -> str:
    if not isinstance(ts, str) or not ts:
        return "n/a"
    try:
        # Support trailing Z
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        return dt.date().isoformat()
    except Exception:
        return ts


def _trade_strategy_map(trades_path: str, window: int = 500) -> Dict[str, str]:
    """Map trade_id -> strategy from the most recent trades.

    Window defaults to last 500 lines to keep it light.
    """
    strat_by_id: Dict[str, str] = {}
    for r in _read_jsonl_tail(trades_path, window):
        tid = r.get("trade_id")
        strat = r.get("strategy")
        if isinstance(tid, str) and tid and isinstance(strat, str) and strat:
            strat_by_id[tid] = strat
    return strat_by_id


def get_top_confidence_suggestions(feedback_path: str, trades_path: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top suggestions by absolute delta, joined with strategy via trades.log.

    Output fields: strategy, old_confidence, new_confidence, delta, date, reason.
    """
    rows = _read_jsonl(feedback_path)
    strat_map = _trade_strategy_map(trades_path)

    items: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        cur = _to_float(r.get("current_confidence") or r.get("confidence"))
        new = _to_float(r.get("suggested_confidence") or r.get("suggested_value") or r.get("value"))
        if cur is None or new is None:
            continue
        delta = new - cur
        strategy = strat_map.get(r.get("trade_id")) or r.get("strategy") or r.get("strategy_name") or "Unknown"
        items.append(
            (
                abs(delta),
                {
                    "strategy": strategy,
                    "old_confidence": round(cur, 4),
                    "new_confidence": round(new, 4),
                    "delta": round(delta, 4),
                    "date": _iso_date(r.get("timestamp")),
                    "reason": r.get("reason", ""),
                },
            )
        )

    items.sort(key=lambda x: (-x[0],))
    return [d for _, d in items[:top_n]]


def format_confidence_suggestions_markdown(items: List[Dict[str, Any]]) -> str:
    """Return markdown list for Streamlit rendering.

    Example usage:
        import streamlit as st
        from src.crypto_trading_bot.analytics.learning_summary import (
            get_top_confidence_suggestions, format_confidence_suggestions_markdown,
        )
        data = get_top_confidence_suggestions("logs/learning_feedback.jsonl", "logs/trades.log")
        st.markdown(format_confidence_suggestions_markdown(data))
    """
    if not items:
        return "✅ No confidence suggestions found."
    lines = ["✅ Top Confidence Suggestions:"]
    for s in items:
        strat = s.get("strategy", "Unknown")
        old_c = s.get("old_confidence")
        new_c = s.get("new_confidence")
        date = s.get("date", "n/a")
        reason = (s.get("reason") or "").strip()
        lines.append(f"- {strat}: {old_c:.3f} → {new_c:.3f} • {date}" + (f" — {reason}" if reason else ""))
    return "\n".join(lines)


def get_top_shadow_tests(shadow_path: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Return top shadow tests by win_rate (desc).

    Tries multiple field names to be robust to format differences.
    Output fields: strategy, win_rate, duration, status.
    """
    rows = _read_jsonl(shadow_path)
    items: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        wr = _to_float(r.get("win_rate") or r.get("success_rate"))
        if wr is None:
            continue

        # Strategy name
        strategy = r.get("strategy") or r.get("strategy_name") or "Unknown"

        # Duration: prefer explicit fields, else derive or mark n/a
        duration = r.get("duration") or r.get("test_duration")
        if duration is None:
            # If start/end timestamps exist, compute rough duration
            t0 = r.get("start") or r.get("start_time") or r.get("timestamp_start")
            t1 = r.get("end") or r.get("end_time") or r.get("timestamp_end")
            duration = "n/a"
            try:
                if isinstance(t0, str) and isinstance(t1, str):
                    a = datetime.fromisoformat(t0.replace("Z", "+00:00"))
                    b = datetime.fromisoformat(t1.replace("Z", "+00:00"))
                    duration = str(b - a)
            except Exception:
                duration = "n/a"

        # Status: use field if present; otherwise derive with a simple threshold
        status = r.get("status")
        if not isinstance(status, str):
            status = "pass" if wr >= 0.5 else "fail"

        items.append(
            (
                wr,
                {
                    "strategy": strategy,
                    "win_rate": round(wr, 4),
                    "duration": duration,
                    "status": status,
                },
            )
        )

    items.sort(key=lambda x: (-x[0],))
    return [d for _, d in items[:top_n]]


def format_shadow_tests_markdown(items: List[Dict[str, Any]]) -> str:
    """Return markdown list for Streamlit rendering of shadow tests.

    Example:
        st.markdown(format_shadow_tests_markdown(get_top_shadow_tests("logs/shadow_test_results.jsonl")))
    """
    if not items:
        return "📊 No shadow test results."
    lines = ["📊 Top Shadow Tests:"]
    for r in items:
        strat = r.get("strategy", "Unknown")
        wr = r.get("win_rate")
        status = (r.get("status") or "").lower()
        duration = r.get("duration", "n/a")
        emoji = "✅" if status == "pass" else "❌"
        lines.append(f"- {emoji} {strat}: {wr*100:.2f}% WR • {status} • {duration}")
    return "\n".join(lines)


def get_recent_anomalies(path: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Return most recent anomalies with type, trade_id, timestamp.

    The anomalies logger writes message-only JSON lines.
    """
    rows = _read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for r in rows[-limit:]:
        out.append(
            {
                "type": r.get("type", "unknown"),
                "trade_id": r.get("trade_id", "n/a"),
                "timestamp": r.get("timestamp") or r.get("time") or "",
            }
        )
    return out[::-1]


def format_anomalies_markdown(items: List[Dict[str, Any]]) -> str:
    """Return markdown list for Streamlit rendering of anomalies.

    Example:
        st.markdown(format_anomalies_markdown(get_recent_anomalies("logs/anomalies.log")))
    """
    if not items:
        return "⚠️ No anomalies found."
    lines = ["⚠️ Recent Anomalies:"]
    for a in items:
        t = a.get("type", "unknown")
        tid = a.get("trade_id", "n/a")
        ts = a.get("timestamp", "")
        lines.append(f"- {t} — {tid} — {ts}")
    return "\n".join(lines)


def get_strategy_leaderboard(trades_path: str, window: int = 30, top_n: int = 5) -> List[Dict[str, Any]]:
    """Rank strategies by ROI and win rate over last `window` closed trades.

    Output fields: strategy, trades, win_rate, avg_roi.
    """
    rows = _read_jsonl(trades_path)
    recent = rows[-window:] if rows else []
    # Use only closed trades for outcome metrics
    closed = [r for r in recent if (r.get("status") or "").lower() == "closed"]

    by: Dict[str, List[float]] = {}
    for r in closed:
        strat = r.get("strategy") or "Unknown"
        roi = _to_float(r.get("roi"))
        if roi is None:
            continue
        by.setdefault(strat, []).append(roi)

    leaderboard: List[Tuple[float, float, int, str]] = []  # (-score, win_rate, n, strat)
    rows_fmt: List[Dict[str, Any]] = []
    for strat, rois in by.items():
        if not rois:
            continue
        n = len(rois)
        wins = sum(1 for r in rois if r > 0)
        win_rate = wins / n
        avg_roi = sum(rois) / n
        # Score: combine win_rate (70%) and avg_roi (30%) to rank
        score = 0.7 * win_rate + 0.3 * max(avg_roi, -1.0)
        leaderboard.append((-score, win_rate, n, strat))
    leaderboard.sort()

    for _neg, win_rate, n, strat in leaderboard[:top_n]:
        rois = by[strat]
        avg_roi = sum(rois) / len(rois) if rois else 0.0
        rows_fmt.append(
            {
                "strategy": strat,
                "trades": n,
                "win_rate": round(win_rate, 4),
                "avg_roi": round(avg_roi, 4),
            }
        )

    return rows_fmt


def format_leaderboard_markdown(items: List[Dict[str, Any]]) -> str:
    """Return markdown list for Streamlit rendering of strategy leaderboard.

    Example:
        rows = get_strategy_leaderboard("logs/trades.log", window=30, top_n=5)
        st.markdown(format_leaderboard_markdown(rows))
    """
    if not items:
        return "📈 No closed trades in the selected window."
    lines = ["📈 Strategy Leaderboard (Last 30):"]
    for i, r in enumerate(items, start=1):
        strat = r.get("strategy", "Unknown")
        wr = r.get("win_rate", 0.0) * 100
        avg = r.get("avg_roi", 0.0) * 100
        n = r.get("trades", 0)
        lines.append(f"- #{i} {strat} — WR {wr:.2f}%, Avg ROI {avg:.2f}%, {n} trades")
    return "\n".join(lines)


# Convenience helpers that compute and format in one call


def get_confidence_suggestions_md(feedback_path: str, trades_path: str, top_n: int = 5) -> str:
    return format_confidence_suggestions_markdown(
        get_top_confidence_suggestions(feedback_path, trades_path, top_n=top_n)
    )


def get_shadow_tests_md(shadow_path: str, top_n: int = 3) -> str:
    return format_shadow_tests_markdown(get_top_shadow_tests(shadow_path, top_n=top_n))


def get_anomalies_md(path: str, limit: int = 3) -> str:
    return format_anomalies_markdown(get_recent_anomalies(path, limit=limit))


def get_leaderboard_md(trades_path: str, window: int = 30, top_n: int = 5) -> str:
    return format_leaderboard_markdown(get_strategy_leaderboard(trades_path, window=window, top_n=top_n))


__all__ = [
    # Raw data functions
    "get_top_confidence_suggestions",
    "get_top_shadow_tests",
    "get_recent_anomalies",
    "get_strategy_leaderboard",
    # Markdown formatters
    "format_confidence_suggestions_markdown",
    "format_shadow_tests_markdown",
    "format_anomalies_markdown",
    "format_leaderboard_markdown",
    # Combined helpers
    "get_confidence_suggestions_md",
    "get_shadow_tests_md",
    "get_anomalies_md",
    "get_leaderboard_md",
]
