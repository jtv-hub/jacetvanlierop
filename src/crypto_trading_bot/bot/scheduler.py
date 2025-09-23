"""
Scheduler Module

Handles the scheduling of periodic trading bot tasks like trade evaluation,
daily maintenance, and learning updates.
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone

from crypto_trading_bot.bot.state.portfolio_state import (
    load_portfolio_state,
    refresh_portfolio_state,
)
from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.bot.utils.log_rotation import (
    get_anomalies_logger,
    get_rotating_handler,
)
from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.learning.confidence_audit import (
    run_and_cleanup as audit_run_and_cleanup,
)
from crypto_trading_bot.learning.learning_machine import run_learning_cycle, run_learning_machine
from crypto_trading_bot.learning.optimization import detect_outliers
from crypto_trading_bot.learning.shadow_test_runner import run_shadow_tests
from crypto_trading_bot.scripts.check_exit_conditions import main as run_exit_checks
from crypto_trading_bot.scripts.daily_heartbeat import run_daily_tasks
from crypto_trading_bot.scripts.shadow_confidence_test import run_shadow_confidence_test
from crypto_trading_bot.scripts.suggest_top_configs import (
    export_suggestions,
    generate_parameter_suggestions,
)
from crypto_trading_bot.scripts.sync_validator import SyncValidator

# Constants for task intervals in seconds
TRADE_INTERVAL = 5 * 60  # Every 5 minutes
DAILY_TASK_HOUR = 0  # Midnight UTC
DAILY_TASK_MINUTE = 5  # Buffer to ensure market data is updated
ANOMALY_AUDIT_INTERVAL = 6 * 60 * 60  # 6 hours in seconds
ALERTS_LOG_PATH = "logs/alerts.log"

anomalies_logger = get_anomalies_logger()


def send_alert(message: str, context: dict | None = None, level: str = "ERROR"):
    """Append alerts to logs/alerts.log as JSONL. Future hook for email/webhooks."""
    try:
        os.makedirs("logs", exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        if context:
            payload.update({"context": context})
        with open(ALERTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except (OSError, IOError):
        # Best-effort alerting; ignore failures
        return


def run_anomaly_audit() -> bool:
    """Run audit with cleanup of closed positions; return True if final state passes."""
    try:
        result = audit_run_and_cleanup("logs/trades.log", "logs/positions.jsonl")
        initial_errors = result.get("initial_errors", 0)
        removed = result.get("removed", 0)
        final_errors = result.get("final_errors", 0)
        if final_errors > 0:
            print("⚠️ Audit still failing after cleanup:")
            for err in result.get("errors", []):
                print(f" - {err}")
            send_alert(
                "Anomaly audit failed after cleanup",
                context={
                    "initial_errors": initial_errors,
                    "removed": removed,
                    "final_errors": final_errors,
                },
                level="CRITICAL",
            )
            return False
        else:
            msg = (
                "🧹 Audit cleanup complete — " f"initial_errors={initial_errors}, removed={removed}, " "final_errors=0"
            )
            print(msg)
            return True
    except (OSError, IOError, ValueError, KeyError, RuntimeError) as e:
        print(f"[Scheduler] run_anomaly_audit failed: {e}")
        send_alert("run_anomaly_audit failed", context={"error": str(e)})
        return False


def update_shadow_test_results():
    """Append a slippage-adjusted summary based on latest closed trades in trades.log.

    We compute win_rate as fraction of closed trades with realized_gain > 0. Assumes ledger already
    accounts for slippage in entry/exit. This supplements per-cycle stats.
    """
    try:
        path = "logs/trades.log"
        if not os.path.exists(path):
            return
        closed = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                    if t.get("status") == "closed" and t.get("exit_price") is not None:
                        closed.append(t)
                except json.JSONDecodeError:
                    continue
        if not closed:
            return
        wins = sum(1 for t in closed if (t.get("realized_gain") or 0) > 0)
        num_exits = len(closed)
        win_rate = wins / num_exits if num_exits else 0.0
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "win_rate": win_rate,
            "num_exits": num_exits,
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/shadow_test_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
    except (OSError, IOError, ValueError) as e:
        send_alert("update_shadow_test_results failed", context={"error": str(e)})
        # Non-fatal
        return


def run_daily_pipeline() -> None:
    """Run all daily tasks: heartbeat, optimization, shadow testing, learning."""
    state = refresh_portfolio_state()
    available = float(state.get("available_capital", 0.0) or 0.0)
    print("🧹 Rotating logs before running daily tasks...")
    print(f"💰 Portfolio available capital: ${available:,.2f}")
    print("\n🌅 Running daily heartbeat tasks...")
    run_daily_tasks()

    print("🧠 Running shadow optimization suggestions...")
    top_configs = detect_outliers(min_trades=25, top_n=3)
    if top_configs:
        suggestions = generate_parameter_suggestions(top_configs)
        export_suggestions(suggestions)
        print("✅ Optimization suggestions complete.")

        print("🧪 Running shadow test evaluation...")
        try:
            run_shadow_tests(output_file="logs/shadow_test_results.jsonl")
            print("📄 Shadow test results saved to logs/shadow_test_results.jsonl.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "module": "scheduler.run_daily_pipeline",
                "action": "run_shadow_tests",
                "message": "Shadow tests execution failed",
                "error": str(exc),
            }
            anomalies_logger.info(json.dumps(error_payload, separators=(",", ":")))
            print(f"[Scheduler] run_shadow_tests failed: {exc}")
    else:
        print("⚠️ No top configurations found for suggestion. Skipping shadow tests.")

    # Emit learning suggestions for dashboard consumption
    try:
        wrote = run_learning_machine()
        print(f"✍️  Learning suggestions written: {wrote}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": "scheduler.run_daily_pipeline",
            "action": "run_learning_machine",
            "message": "Learning machine execution failed",
            "error": str(exc),
        }
        anomalies_logger.info(json.dumps(error_payload, separators=(",", ":")))
        print(f"[Scheduler] run_learning_machine failed: {exc}")

    metrics = run_learning_cycle()
    print("📊 Learning Summary:", metrics)

    # Confidence threshold analysis (append-only diagnostics; no prod effect)
    try:
        n_rows = run_shadow_confidence_test()
        print(f"📐 Confidence threshold analysis appended {n_rows} row(s).")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Scheduler] run_shadow_confidence_test failed: {e}")


def should_run_daily(last_run_time):
    """Check if a new daily task run is due."""
    now = datetime.now(timezone.utc)
    return (
        now.hour == DAILY_TASK_HOUR
        and now.minute >= DAILY_TASK_MINUTE
        and (last_run_time is None or last_run_time.date() < now.date())
    )


def should_run_anomaly_audit(last_audit_time):
    """Check if the anomaly audit should run based on time interval."""
    if last_audit_time is None:
        return True
    return (datetime.now(timezone.utc) - last_audit_time).total_seconds() >= ANOMALY_AUDIT_INTERVAL


def run_scheduler():
    """Runs the main scheduler loop that handles trade evaluation and daily bot maintenance."""
    print("📅 Scheduler started. Running bot tasks...")

    get_rotating_handler("trades.log")
    get_rotating_handler("anomalies.log")
    get_rotating_handler("shadow_test_results.jsonl")
    learning_metrics = run_learning_cycle()
    buffer_pct = learning_metrics.get("capital_buffer", 0.0)

    # Set default adjusted risk before conditions
    adjusted_risk = 0.02

    if buffer_pct > 0.25:
        adjusted_risk = 0.02 * 0.5
        print(f"🛡️ Buffer high ({round(buffer_pct * 100)}%), " f"risk ↓ to {adjusted_risk * 100:.1f}%")
    elif buffer_pct > 0.10:
        adjusted_risk = 0.02 * 0.75
        buffer_msg = f"⚠️ Buffer elevated ({round(buffer_pct * 100)}%), " f"risk ↓ to {adjusted_risk * 100:.1f}%"
        print(buffer_msg)
    else:
        print(f"✅ Capital buffer low ({round(buffer_pct * 100)}%)")
        print("Using full risk allocation")

    last_daily_run = None
    last_audit_run = None

    portfolio_state = load_portfolio_state(refresh=True)

    # Kick off at least one suggestion write so dashboards have data on first run
    try:
        wrote_boot = run_learning_machine()
        if wrote_boot:
            print(f"✍️  Boot suggestions written: {wrote_boot}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": "scheduler.run_scheduler",
            "action": "run_learning_machine_initial",
            "message": "Initial learning machine execution failed",
            "error": str(exc),
        }
        anomalies_logger.info(json.dumps(error_payload, separators=(",", ":")))
        print(f"[Scheduler] Initial run_learning_machine failed: {exc}")

    while True:
        try:
            # Refresh portfolio state and run trade evaluation
            portfolio_state = load_portfolio_state(refresh=True)
            available_capital = float(portfolio_state.get("available_capital", 0.0))
            reinvestment_rate = float(portfolio_state.get("reinvestment_rate", 0.0))

            if available_capital <= 0:
                print("⚠️ Available capital is non-positive — skipping trade evaluation.")
            else:
                print("\n⏱️ Evaluating trades...")
                evaluate_signals_and_trade(
                    tradable_pairs=CONFIG.get("tradable_pairs", []),
                    available_capital=available_capital,
                    risk_per_trade=adjusted_risk,
                    reinvestment_rate=reinvestment_rate,
                )

            print("🔁 Checking exit conditions...")
            run_exit_checks()

            # Run sync validation each cycle after exits
            try:
                validator = SyncValidator()
                ok = validator.validate_sync()
                if not ok:
                    print("⚠️ Sync validation issues detected:")
                    for err in validator.validation_errors:
                        print(f" - {err}")
                else:
                    print("✅ Sync validation passed.")
            except (ValueError, RuntimeError, OSError) as e:
                print(f"[Scheduler] SyncValidator failed: {e}")

            # Run anomaly audit every 6 hours
            if should_run_anomaly_audit(last_audit_run):
                print("🧹 Running anomaly audit...")
                ok = run_anomaly_audit()
                last_audit_run = datetime.now(timezone.utc)
                if not ok:
                    # Halt the scheduler on persistent audit failures
                    raise SystemExit("Audit failed after cleanup; halting scheduler.")

            # Supplemental shadow testing summary (slippage-adjusted)
            update_shadow_test_results()

            # Run daily tasks once per UTC day
            if should_run_daily(last_daily_run):
                run_daily_pipeline()
                last_daily_run = datetime.now(timezone.utc)

            time.sleep(TRADE_INTERVAL)

        except KeyboardInterrupt:
            print("🛑 Scheduler stopped by user.")
            break
        except (ValueError, RuntimeError, OSError) as error:
            print(f"❌ Scheduler error: {error}")
            traceback.print_exc()


if __name__ == "__main__":
    run_scheduler()
