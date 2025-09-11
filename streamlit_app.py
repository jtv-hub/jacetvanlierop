"""
Streamlit dashboard: Top Strategies Leaderboard.

Displays number of trades, win rate, and average ROI per strategy
from logs/trades.log.
"""

from __future__ import annotations

import json
import os
from typing import List

import pandas as pd
import streamlit as st

LOG_PATH = "logs/trades.log"


def _load_trades(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("status") or "").lower() in {"executed", "closed"}:
                rows.append(rec)
    return rows


def _make_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Consider only closed trades with numeric ROI for accuracy
    closed = df[(df["status"].str.lower() == "closed") & df["roi"].apply(lambda x: isinstance(x, (int, float)))]
    if closed.empty:
        return pd.DataFrame(columns=["strategy", "trades", "win_rate", "avg_roi"])
    grp = closed.groupby("strategy", dropna=False)
    trades = grp.size().rename("trades")
    win_rate = grp["roi"].apply(lambda s: (s > 0).mean()).rename("win_rate")
    avg_roi = grp["roi"].mean().rename("avg_roi")
    out = pd.concat([trades, win_rate, avg_roi], axis=1).reset_index()
    out = out.sort_values(["avg_roi", "win_rate"], ascending=[False, False])
    out["win_rate"] = (out["win_rate"] * 100).round(2)
    out["avg_roi"] = out["avg_roi"].round(4)
    return out


def main() -> None:
    st.set_page_config(page_title="Crypto Bot Dashboard", layout="wide")
    st.title("Crypto Trading Bot Dashboard")

    trades = _load_trades(LOG_PATH)
    df = pd.DataFrame(trades) if trades else pd.DataFrame()
    for col in ("strategy", "status"):
        if col not in df.columns:
            df[col] = None

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Top Performing Strategies",
            "Equity Curve",
            "Last 10 Trades",
            "Learning Suggestions",
            "Shadow Tests",
        ]
    )

    with tab1:
        st.header("Top Performing Strategies")
        if df.empty:
            st.info("No trades found in logs/trades.log yet.")
        else:
            lb = _make_leaderboard(df)
            st.dataframe(lb, use_container_width=True)

    with tab2:
        st.header("Equity Curve (ROI cumulative)")
        if not df.empty and "roi" in df.columns:
            closed = df[(df["status"].str.lower() == "closed") & df["roi"].apply(lambda x: isinstance(x, (int, float)))]
            if not closed.empty:
                closed = closed.sort_values("timestamp")
                closed["equity"] = (1 + closed["roi"]).cumprod()
                st.line_chart(closed.set_index("timestamp")["equity"])
            else:
                st.info("No closed trades with ROI yet.")
        else:
            st.info("No ROI data available.")

    with tab3:
        st.header("Last 10 Trades")
        if not df.empty:
            cols = ["timestamp", "pair", "strategy", "side", "roi", "reason"]
            cols = [c for c in cols if c in df.columns]
            st.table(df.tail(10)[cols])
        else:
            st.info("No trades yet.")

    with tab4:
        st.header("Learning Suggestions")
        lf_path = "logs/learning_feedback.jsonl"
        sugg = _load_trades(lf_path)
        if sugg:
            st.dataframe(pd.DataFrame(sugg), use_container_width=True)
        else:
            st.info("No learning feedback yet.")

    with tab5:
        st.header("Shadow Test Results")
        st_path = "logs/shadow_test_results.jsonl"
        sh = _load_trades(st_path)
        if sh:
            st.dataframe(pd.DataFrame(sh), use_container_width=True)
        else:
            st.info("No shadow test results logged yet.")


if __name__ == "__main__":
    main()
