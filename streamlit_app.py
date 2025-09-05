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
    # Closed Trades Overview (JSONL -> DataFrame)
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

    # Build a DataFrame for convenient analysis
    df_closed = None
    if closed_sorted:
        df_closed = pd.DataFrame(
            [
                {
                    "trade_id": t.get("trade_id"),
                    "timestamp": t.get("timestamp"),
                    "strategy": t.get("strategy"),
                    "roi": float(t.get("roi", 0.0)),
                    "reason": t.get("reason") or t.get("exit_reason"),
                    "exit_price": t.get("exit_price"),
                }
                for t in closed_sorted
            ]
        )
        # Parse timestamp to datetime for proper sorting/plotting
        try:
            df_closed["timestamp"] = pd.to_datetime(df_closed["timestamp"], errors="coerce")
        except Exception:
            pass

    st.header("📊 Closed Trades Overview")
    if df_closed is not None and not df_closed.empty:
        # Last 10 closed trades table
        last10 = df_closed.sort_values("timestamp").tail(10)
        st.subheader("Last 10 Closed Trades")
        st.dataframe(
            last10[["trade_id", "timestamp", "strategy", "roi", "reason", "exit_price"]], use_container_width=True
        )

        # Running ROI (cumulative) and Win Rate
        try:
            rois = df_closed["roi"].astype(float).fillna(0.0)
            win_rate = (rois.gt(0).sum() / len(rois)) if len(rois) else 0.0
            cum_curve = (1.0 + rois).cumprod() - 1.0
            cumulative_roi = float(cum_curve.iloc[-1]) if not cum_curve.empty else 0.0
        except Exception:
            win_rate, cumulative_roi = 0.0, 0.0

        c1, c2 = st.columns(2)
        c1.metric("Running Win Rate", f"{win_rate*100:.2f}%")
        c2.metric("Cumulative ROI", f"{cumulative_roi*100:.2f}%")

        # Cumulative ROI line chart over time
        st.subheader("Cumulative ROI Over Time")
        try:
            plot_df = df_closed.sort_values("timestamp").copy()
            plot_df["cum_roi"] = (1.0 + plot_df["roi"].astype(float)).cumprod() - 1.0
            plot_df = plot_df.set_index("timestamp")
            st.line_chart(plot_df[["cum_roi"]])
        except Exception:
            st.info("Not enough data to plot cumulative ROI.")

        # Most common exit reason
        try:
            common_reason = (
                df_closed["reason"].dropna().astype(str).str.strip().value_counts().idxmax()
                if not df_closed["reason"].dropna().empty
                else "n/a"
            )
        except Exception:
            common_reason = "n/a"

        # Most profitable strategy by average ROI
        try:
            strat_avg = (
                df_closed.groupby("strategy")["roi"].mean().sort_values(ascending=False)
                if "strategy" in df_closed.columns and not df_closed.empty
                else None
            )
            top_strat = strat_avg.index[0] if strat_avg is not None and len(strat_avg) > 0 else "n/a"
            top_strat_avg = float(strat_avg.iloc[0]) if strat_avg is not None and len(strat_avg) > 0 else 0.0
        except Exception:
            top_strat, top_strat_avg = "n/a", 0.0

        c3, c4 = st.columns(2)
        c3.metric("Most Common Exit Reason", common_reason)
        c4.metric("Top Strategy (Avg ROI)", f"{top_strat}: {top_strat_avg*100:.2f}%")
    else:
        st.info("No closed trades found in logs/trades.log yet.")

    # ----------------------------
    # Equity Curve / Running Balance (legacy view)
    # ----------------------------

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
    st.header("📄 Closed Trades (Full) — Legacy Table")
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
        st.dataframe(pd.DataFrame(table), use_container_width=True)
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

    # ----------------------------
    # Confidence Threshold Analysis
    # ----------------------------
    st.header("📐 Confidence Threshold Analysis")
    # Load recent confidence-threshold evaluations
    threshold_rows = []
    if os.path.exists(shadow_path):
        try:
            with open(shadow_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "confidence_threshold_eval":
                        threshold_rows.append(
                            {
                                "threshold": obj.get("threshold"),
                                "avg_roi": obj.get("avg_roi"),
                                "win_rate": obj.get("win_rate"),
                                "cumulative_roi": obj.get("cumulative_roi"),
                            }
                        )
        except OSError as e:
            st.warning(f"Could not read shadow_test_results.jsonl: {e}")

    if threshold_rows and pd is not None:
        df_th = pd.DataFrame(threshold_rows)
        # Coerce numeric and drop invalid
        for col in ("threshold", "avg_roi", "win_rate", "cumulative_roi"):
            df_th[col] = pd.to_numeric(df_th[col], errors="coerce")
        df_th = df_th.dropna(subset=["threshold"]).sort_values("threshold")
        if not df_th.empty:
            chart_df = df_th.set_index("threshold")[["avg_roi", "win_rate"]]
            st.line_chart(chart_df)

            # Simple annotations for optimal points
            try:
                best_avg = chart_df["avg_roi"].idxmax()
                best_wr = chart_df["win_rate"].idxmax()
                st.caption(f"Best avg ROI at threshold {best_avg:.1f}; " f"best win rate at threshold {best_wr:.1f}.")
            except Exception:
                pass
        else:
            st.info("No shadow confidence test results yet.")
    else:
        st.info("No shadow confidence test results yet.")

    st.header("⚠️ Anomalies")
    st.subheader("⚠️ Recent Anomalies")
    st.markdown(get_anomalies_md(anomalies_path, limit=3))

    st.header("🏆 Strategy Leaderboard")
    st.subheader("📈 Strategy Leaderboard (Last 30 Trades)")
    st.markdown(get_leaderboard_md(trades_path, window=30, top_n=5))

    # ----------------------------
    # Learning Machine Summary
    # ----------------------------
    st.header("📚 Learning Machine Summary")
    lf_path = feedback_path
    df_lf = None
    if os.path.exists(lf_path) and pd is not None:
        try:
            # Prefer pandas JSONL reader
            df_lf = pd.read_json(lf_path, lines=True)
        except ValueError:
            # Fallback: best-effort manual load then frame
            rows = []
            with open(lf_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if rows:
                df_lf = pd.DataFrame(rows)

    if df_lf is None or df_lf.empty:
        st.info("No learning feedback entries yet.")
    else:
        # Infer missing 'type' where possible for compatibility
        if "type" not in df_lf.columns:
            df_lf["type"] = None
        # Infer no_suggestions entries from status value
        mask_no = df_lf.get("status").astype(str).str.lower().eq("no_suggestions_generated")
        df_lf.loc[mask_no, "type"] = df_lf.loc[mask_no, "type"].fillna("no_suggestions_generated")
        # Infer learning suggestions if suggestion-like fields present
        has_suggestion_fields = (
            (df_lf.get("suggestion").notna() if "suggestion" in df_lf.columns else False)
            | (df_lf.get("suggested_confidence").notna() if "suggested_confidence" in df_lf.columns else False)
            | (df_lf.get("current_confidence").notna() if "current_confidence" in df_lf.columns else False)
        )
        if hasattr(has_suggestion_fields, "any"):
            df_lf.loc[has_suggestion_fields, "type"] = df_lf.loc[has_suggestion_fields, "type"].fillna(
                "learning_suggestion"
            )

        # Filter to the requested types only
        df_keep = df_lf[df_lf["type"].isin(["learning_suggestion", "no_suggestions_generated"])].copy()
        if df_keep.empty:
            st.info("No learning suggestion or diagnostic entries found.")
        else:
            # Total Suggestions Summary
            df_sug = df_keep[df_keep["type"] == "learning_suggestion"].copy()
            df_no = df_keep[df_keep["type"] == "no_suggestions_generated"].copy()

            total_suggestions = int(len(df_sug))

            def _count_status(val: str) -> int:
                return (
                    int(df_sug.get("status").astype(str).str.lower().eq(val).sum()) if "status" in df_sug.columns else 0
                )

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Suggestions", f"{total_suggestions}")
            col_b.metric("Approved", f"{_count_status('approved')}")
            col_c.metric("Rejected", f"{_count_status('rejected')}")
            col_d.metric("Shadow Tested", f"{_count_status('shadow_tested')}")

            st.caption(f"No-suggestion diagnostics: {len(df_no)}")

            # Recent Suggestions Table (last 10)
            if not df_sug.empty:
                # Normalize confidence fields
                if "confidence_before" not in df_sug.columns:
                    df_sug["confidence_before"] = df_sug.get("current_confidence")
                if "confidence_after" not in df_sug.columns:
                    df_sug["confidence_after"] = df_sug.get("suggested_confidence")
                # Normalize param/value fields
                if "parameter" not in df_sug.columns:
                    # Try to extract from generic fields; else None
                    df_sug["parameter"] = None
                if "old_value" not in df_sug.columns:
                    df_sug["old_value"] = None
                if "new_value" not in df_sug.columns:
                    df_sug["new_value"] = None

                # Sort by timestamp if present
                if "timestamp" in df_sug.columns:
                    try:
                        df_sug = df_sug.sort_values("timestamp")
                    except Exception:
                        pass

                st.subheader("Recent Suggestions")
                cols = [
                    "timestamp",
                    "strategy",
                    "parameter",
                    "old_value",
                    "new_value",
                    "confidence_before",
                    "confidence_after",
                    "status",
                ]
                view_cols = [c for c in cols if c in df_sug.columns]
                st.dataframe(df_sug[view_cols].tail(10), use_container_width=True)

                # Confidence Impact Chart (approved only)
                try:
                    df_app = df_sug[df_sug.get("status").astype(str).str.lower().eq("approved")]
                    if not df_app.empty:
                        df_ci = df_app[["timestamp", "confidence_before", "confidence_after"]].copy()
                        # Prepare diff from whatever fields are available
                        df_ci["confidence_before"] = pd.to_numeric(df_ci["confidence_before"], errors="coerce")
                        df_ci["confidence_after"] = pd.to_numeric(df_ci["confidence_after"], errors="coerce")
                        df_ci["delta"] = df_ci["confidence_after"] - df_ci["confidence_before"]
                        df_ci = df_ci.dropna(subset=["delta"])  # keep valid diffs
                        if not df_ci.empty:
                            st.subheader("Confidence Impact (Approved)")
                            df_ci_plot = df_ci.set_index("timestamp")["delta"]
                            st.bar_chart(df_ci_plot)
                except Exception:
                    pass

            # Suggestion Reason Breakdown
            try:
                if "reason" in df_sug.columns and not df_sug["reason"].dropna().empty:
                    st.subheader("Suggestion Reason Breakdown")
                    reason_counts = df_sug["reason"].astype(str).str.strip().value_counts().sort_values(ascending=True)
                    st.bar_chart(reason_counts)
            except Exception:
                pass

    st.caption("Data updates as logs grow. Designed for mobile with markdown sections.")


if __name__ == "__main__":
    main()
