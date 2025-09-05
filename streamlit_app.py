from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable on Streamlit Cloud (prioritize via insert)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.crypto_trading_bot.analytics.learning_summary import (
    get_anomalies_md,
    get_confidence_suggestions_md,
    get_leaderboard_md,
    get_shadow_tests_md,
)
from src.crypto_trading_bot.analytics.roi_calculator import compute_running_balance


def main() -> None:
    st.set_page_config(page_title="Learning Machine Summary", layout="wide")
    # Always clear caches so the dashboard reflects live files
    try:
        st.cache_data.clear()
    except Exception:
        # Safe-guard: cache API may differ across versions
        pass
    st.set_option("client.showErrorDetails", True)
    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60 * 1000, key="learning_dashboard_autorefresh")
    st.title("🧠 Learning Machine Summary")

    # Log paths
    base_dir = Path(__file__).resolve().parent
    feedback_path = str(base_dir / "logs" / "learning_feedback.jsonl")
    shadow_path = str(base_dir / "logs" / "shadow_test_results.jsonl")
    anomalies_path = str(base_dir / "logs" / "anomalies.log")
    trades_path = str(base_dir / "logs" / "trades.log")

    # ----------------------------
    # Equity Curve / Running Balance
    # ----------------------------
    st.header("📈 Equity Curve / Running Balance")

    # Load trades once for multiple sections
    trades_rows: list[dict] = []
    total_lines = 0
    malformed = 0
    if os.path.exists(trades_path):
        try:
            with open(trades_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1
                    try:
                        trades_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
        except OSError as e:
            st.warning(f"Could not read trades log: {e}")
    else:
        st.warning("logs/trades.log not found. The dashboard will be empty until trades are logged.")

    # Filter closed trades and sort by timestamp
    def _parse_ts(ts: str | None) -> str:
        if not isinstance(ts, str) or not ts:
            return ""
        return ts

    # Validate essential fields for closed trades
    def _is_valid_closed(t: dict) -> bool:
        if (t.get("status") or "").lower() != "closed":
            return False
        try:
            _ = float(t.get("roi"))
            _ = float(t.get("exit_price"))
        except (TypeError, ValueError):
            return False
        ts = t.get("timestamp")
        return isinstance(ts, str) and bool(ts.strip())

    closed_valid = [t for t in trades_rows if _is_valid_closed(t)]
    closed_sorted = sorted(closed_valid, key=lambda r: _parse_ts(r.get("timestamp")))

    # Warn about empty or malformed logs
    if total_lines == 0 and not trades_rows:
        st.warning("trades.log is empty. No trades to display yet.")
    elif malformed > 0:
        st.warning(f"Skipped {malformed} malformed line(s) in trades.log while loading.")

    # Build running balance series (start 1000 by default)
    start_balance = 1000.0
    bal = start_balance
    xs: list[int] = []
    ys: list[float] = []
    for i, t in enumerate(closed_sorted, start=1):
        try:
            size = float(t.get("size"))
            roi = float(t.get("roi"))
        except (TypeError, ValueError):
            continue
        bal += size * roi
        xs.append(i)
        ys.append(round(bal, 2))

    # Also compute final balance via the shared utility
    try:
        final_balance = compute_running_balance(trades_rows, starting_balance=start_balance)
    except Exception:  # safety: never break dashboard
        final_balance = ys[-1] if ys else start_balance

    if ys:
        df_equity = pd.DataFrame({"Balance": ys}, index=xs)
        df_equity.index.name = "Trade #"
        st.line_chart(df_equity)
        st.markdown(f"Final Balance: ${final_balance:,.2f}")
    else:
        st.info("No closed trades yet to plot equity curve.")

    # ----------------------------
    # Table of Closed Trades
    # ----------------------------
    st.header("📄 Closed Trades")
    if closed_sorted:
        cols = [
            "trade_id",
            "pair",
            "strategy",
            "confidence",
            "entry_price",
            "exit_price",
            "roi",
            "exit_reason",
            "timestamp",
        ]
        table = [{k: r.get(k) for k in cols} for r in closed_sorted]
        df_closed = pd.DataFrame(table)
        st.dataframe(df_closed, use_container_width=True)
    else:
        st.info("No closed trades found in logs/trades.log")

    # ----------------------------
    # Markdown sections from analytics helpers
    # ----------------------------
    st.header("✅ Top Confidence Suggestions")
    st.subheader("✅ Top Confidence Suggestions")
    st.markdown(get_confidence_suggestions_md(feedback_path, trades_path, top_n=5))

    st.header("🧪 Shadow Test Results")
    st.subheader("📊 Top Shadow Tests")
    st.markdown(get_shadow_tests_md(shadow_path, top_n=3))

    st.header("⚠️ Anomalies")
    st.subheader("⚠️ Recent Anomalies")
    st.markdown(get_anomalies_md(anomalies_path, limit=3))

    st.header("🏆 Strategy Leaderboard")
    st.subheader("📈 Strategy Leaderboard (Last 30 Trades)")
    st.markdown(get_leaderboard_md(trades_path, window=30, top_n=5))

    st.caption("Data updates as logs grow. Designed for mobile with markdown sections.")


if __name__ == "__main__":
    main()
