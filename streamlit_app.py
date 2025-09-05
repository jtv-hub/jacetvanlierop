from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable on Streamlit Cloud
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import os

import streamlit as st

from src.crypto_trading_bot.analytics.learning_summary import (
    get_anomalies_md,
    get_confidence_suggestions_md,
    get_leaderboard_md,
    get_shadow_tests_md,
)


def main() -> None:
    st.set_page_config(page_title="Learning Machine Summary", layout="wide")
    st.title("🧠 Learning Machine Summary")

    # Log paths
    feedback_path = os.path.join("logs", "learning_feedback.jsonl")
    shadow_path = os.path.join("logs", "shadow_test_results.jsonl")
    anomalies_path = os.path.join("logs", "anomalies.log")
    trades_path = os.path.join("logs", "trades.log")

    # Sections
    st.subheader("✅ Top Confidence Suggestions")
    st.markdown(get_confidence_suggestions_md(feedback_path, trades_path, top_n=5))

    st.subheader("📊 Top Shadow Tests")
    st.markdown(get_shadow_tests_md(shadow_path, top_n=3))

    st.subheader("⚠️ Recent Anomalies")
    st.markdown(get_anomalies_md(anomalies_path, limit=3))

    st.subheader("📈 Strategy Leaderboard (Last 30 Trades)")
    st.markdown(get_leaderboard_md(trades_path, window=30, top_n=5))

    st.caption("Data updates as logs grow. Designed for mobile with markdown sections.")


if __name__ == "__main__":
    main()
