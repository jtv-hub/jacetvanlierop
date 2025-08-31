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

from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler
from crypto_trading_bot.learning.confidence_audit import (
    run_and_cleanup as audit_run_and_cleanup,
)
from crypto_trading_bot.learning.learning_machine import run_learning_cycle
from crypto_trading_bot.learning.optimization import detect_outliers
from crypto_trading_bot.learning.shadow_test_runner import run_shadow_tests
from crypto_trading_bot.scripts.check_exit_conditions import main as run_exit_checks
from crypto_trading_bot.scripts.daily_heartbeat import run_daily_tasks
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
    except Exception:
        # Best-effort alerting; ignore failures
        pass


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
            print(f"🧹 Audit cleanup complete — initial_errors={initial_errors}, removed={removed}, final_errors=0")
            return True
    except Exception as e:
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
    except Exception as e:
        send_alert("update_shadow_test_results failed", context={"error": str(e)})
        # Non-fatal
        pass


def run_daily_pipeline():
    """Run all daily tasks: heartbeat, optimization, shadow testing, learning."""
    print("🧹 Rotating logs before running daily tasks...")
    print("\n🌅 Running daily heartbeat tasks...")
    run_daily_tasks()

    print("🧠 Running shadow optimization suggestions...")
    top_configs = detect_outliers(min_trades=25, top_n=3)
    if top_configs:
        suggestions = generate_parameter_suggestions(top_configs)
        export_suggestions(suggestions)
        print("✅ Optimization suggestions complete.")

        print("🧪 Running shadow test evaluation...")
        run_shadow_tests()
        print("📄 Shadow test results saved to shadow_test_results.jsonl.")
    else:
        print("⚠️ No top configurations found for suggestion. Skipping shadow tests.")

    metrics = run_learning_cycle()
    print("📊 Learning Summary:", metrics)


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
        print(f"🛡️ Buffer high ({round(buffer_pct * 100)}%), risk ↓ to {adjusted_risk * 100:.1f}%")
    elif buffer_pct > 0.10:
        adjusted_risk = 0.02 * 0.75
        buffer_msg = f"⚠️ Buffer elevated ({round(buffer_pct * 100)}%), " f"risk ↓ to {adjusted_risk * 100:.1f}%"
        print(buffer_msg)
    else:
        print(f"✅ Capital buffer low ({round(buffer_pct * 100)}%)")
        print("Using full risk allocation")

    last_daily_run = None
    last_audit_run = None

    while True:
        try:
            # Run trade evaluation
            print("\n⏱️ Evaluating trades...")
            evaluate_signals_and_trade()

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
            except Exception as e:
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
