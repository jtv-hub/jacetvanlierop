from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable on Streamlit Cloud (prioritize via insert)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from src.crypto_trading_bot.analytics.learning_summary import (
    get_anomalies_md,
    get_confidence_suggestions_md,
    get_leaderboard_md,
    get_shadow_tests_md,
)
from src.crypto_trading_bot.analytics.roi_calculator import compute_running_balance


def main() -> None:
    st.set_page_config(page_title="Learning Machine Summary", layout="wide")
    st.set_option("client.showErrorDetails", True)
    st.title("🧠 Learning Machine Summary")

    # Log paths
    feedback_path = os.path.join("logs", "learning_feedback.jsonl")
    shadow_path = os.path.join("logs", "shadow_test_results.jsonl")
    anomalies_path = os.path.join("logs", "anomalies.log")
    trades_path = os.path.join("logs", "trades.log")

    # ----------------------------
    # Equity Curve / Running Balance
    # ----------------------------
    st.header("📈 Equity Curve / Running Balance")

    # Load trades once for multiple sections
    trades_rows: list[dict] = []
    if os.path.exists(trades_path):
        with open(trades_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trades_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Filter closed trades and sort by timestamp
    def _parse_ts(ts: str | None) -> str:
        if not isinstance(ts, str) or not ts:
            return ""
        return ts

    closed = [t for t in trades_rows if (t.get("status") or "").lower() == "closed"]
    closed_sorted = sorted(closed, key=lambda r: _parse_ts(r.get("timestamp")))

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
