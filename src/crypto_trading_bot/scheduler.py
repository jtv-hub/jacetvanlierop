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
from pathlib import Path

import schedule

from crypto_trading_bot.config import CONFIG, ConfigurationError, get_mode_label, is_live
from crypto_trading_bot.learning.confidence_audit import (
    run_and_cleanup as audit_run_and_cleanup,
)
from crypto_trading_bot.learning.learning_machine import run_learning_cycle, run_learning_machine
from crypto_trading_bot.learning.learning_pipeline import run_learning_pipeline
from crypto_trading_bot.learning.optimization import detect_outliers
from crypto_trading_bot.learning.shadow_test_runner import run_shadow_tests
from crypto_trading_bot.portfolio_state import (
    load_portfolio_state,
    refresh_portfolio_state,
)
from crypto_trading_bot.safety.confirmation import require_live_confirmation
from crypto_trading_bot.scripts.check_exit_conditions import main as run_exit_checks
from crypto_trading_bot.scripts.daily_heartbeat import run_daily_tasks
from crypto_trading_bot.scripts.shadow_confidence_test import run_shadow_confidence_test
from crypto_trading_bot.scripts.suggest_top_configs import (
    export_suggestions,
    generate_parameter_suggestions,
)
from crypto_trading_bot.scripts.sync_validator import SyncValidator
from crypto_trading_bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.utils.log_rotation import (
    get_anomalies_logger,
    get_rotating_handler,
)
from crypto_trading_bot.utils.system_logger import get_system_logger

# Constants for task intervals in seconds
TRADE_INTERVAL = 5 * 60  # Every 5 minutes
DAILY_TASK_HOUR = 0  # Midnight UTC
DAILY_TASK_MINUTE = 5  # Buffer to ensure market data is updated
ANOMALY_AUDIT_INTERVAL = 6 * 60 * 60  # 6 hours in seconds
ALERTS_LOG_PATH = "logs/alerts.log"
CLEAN_CONFIG_TIME = "02:45"

# Daily PPO training (5am UTC)
# 0 5 * * * PYTHONPATH=src venv/bin/python scripts/train_ppo.py --timesteps 50000
# Nightly NSGA-III hyperparameter optimisation (2am UTC)
# 0 2 * * * PYTHONPATH=src venv/bin/python src/crypto_trading_bot/optimization/nsga3_optimizer.py
# TWAP execution smoke check
# 0 5 * * * PYTHONPATH=src venv/bin/python scripts/test_twap.py
print(f"Scheduler: clean_config.py scheduled daily at {CLEAN_CONFIG_TIME} UTC")

anomalies_logger = get_anomalies_logger()
logger = get_system_logger().getChild("scheduler")


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
    if not CONFIG.get("is_live") and not CONFIG.get("test_mode"):
        logger.info("Skipping anomaly audit — live mode disabled.")
        return True
    try:
        result = audit_run_and_cleanup("logs/trades.log", "logs/positions.jsonl")
        initial_errors = result.get("initial_errors", 0)
        removed = result.get("removed", 0)
        final_errors = result.get("final_errors", 0)
        if final_errors > 0:
            logger.warning(
                "Audit still failing after cleanup",
                extra={
                    "initial_errors": initial_errors,
                    "removed": removed,
                    "final_errors": final_errors,
                    "errors": result.get("errors", []),
                },
            )
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
            msg = "🧹 Audit cleanup complete — " f"initial_errors={initial_errors}, removed={removed}, final_errors=0"
            logger.info("Audit cleanup complete", extra={"message": msg})
            return True
    except (OSError, IOError, ValueError, KeyError, RuntimeError) as e:
        logger.error("run_anomaly_audit failed", extra={"error": str(e)})
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


def run_clean_config() -> None:
    """Execute the config cleanup script to deduplicate min_confidence values."""

    script_path = Path("src/crypto_trading_bot/scripts/clean_config.py")
    if not script_path.exists():
        print(f"clean_config.py: missing at {script_path}")
        return
    cmd = "PYTHONPATH=src python src/crypto_trading_bot/scripts/clean_config.py --force"
    result = os.system(cmd)
    if result == 0:
        print("clean_config.py: SUCCESS")
    else:
        print(f"clean_config.py: FAILED (code {result})")


schedule.every().day.at(CLEAN_CONFIG_TIME).do(run_clean_config)
schedule.every(6).hours.do(run_learning_pipeline)


def run_daily_pipeline() -> None:
    """Run all daily tasks: heartbeat, optimization, shadow testing, learning."""
    if not CONFIG.get("is_live") and not CONFIG.get("test_mode"):
        logger.info("Skipping daily pipeline — live mode disabled.")
        return
    state = refresh_portfolio_state()
    available = float(state.get("available_capital", 0.0) or 0.0)
    logger.info("Rotating logs before running daily tasks")
    logger.info("Portfolio available capital", extra={"available_capital": available})
    logger.info("Running daily heartbeat tasks")
    run_daily_tasks()

    logger.info("Running shadow optimization suggestions")
    top_configs = detect_outliers(min_trades=25, top_n=3)
    if top_configs:
        suggestions = generate_parameter_suggestions(top_configs)
        export_suggestions(suggestions)
        logger.info(
            "Optimization suggestions complete",
            extra={"suggestion_count": len(suggestions)},
        )

        logger.info("Running shadow test evaluation")
        try:
            run_shadow_tests(output_file="logs/shadow_test_results.jsonl")
            logger.info(
                "Shadow test results saved",
                extra={"path": "logs/shadow_test_results.jsonl"},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "module": "scheduler.run_daily_pipeline",
                "action": "run_shadow_tests",
                "message": "Shadow tests execution failed",
                "error": str(exc),
            }
            anomalies_logger.info(json.dumps(error_payload, separators=(",", ":")))
            logger.error("run_shadow_tests failed during daily pipeline", extra={"error": str(exc)})
    else:
        logger.warning("No top configurations found for suggestion; skipping shadow tests.")

    # Emit learning suggestions for dashboard consumption
    try:
        wrote = run_learning_machine()
        logger.info("Learning suggestions written", extra={"count": wrote})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": "scheduler.run_daily_pipeline",
            "action": "run_learning_machine",
            "message": "Learning machine execution failed",
            "error": str(exc),
        }
        anomalies_logger.info(json.dumps(error_payload, separators=(",", ":")))
        logger.error("run_learning_machine failed during daily pipeline", extra={"error": str(exc)})

    metrics = run_learning_cycle()
    logger.info("Learning summary", extra={"metrics": metrics})

    # Confidence threshold analysis (append-only diagnostics; no prod effect)
    try:
        n_rows = run_shadow_confidence_test()
        logger.info(
            "Confidence threshold analysis appended rows",
            extra={"rows_appended": n_rows},
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("run_shadow_confidence_test failed", extra={"error": str(e)})


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
    logger.info("Scheduler started; running bot tasks")
    paper_mode = not CONFIG.get("is_live") and not CONFIG.get("test_mode")
    if paper_mode:
        logger.warning("Live mode disabled in configuration — running scheduler in paper simulation mode.")
    mode_label = get_mode_label()
    logger.info("Operating mode resolved", extra={"mode": mode_label, "is_live": is_live})

    get_rotating_handler("trades.log")
    get_rotating_handler("anomalies.log")
    get_rotating_handler("shadow_test_results.jsonl")
    learning_metrics = run_learning_cycle()
    buffer_pct = learning_metrics.get("capital_buffer", 0.0)

    # Set default adjusted risk before conditions
    adjusted_risk = 0.02

    if buffer_pct > 0.25:
        adjusted_risk = 0.02 * 0.5
        logger.info(
            "Capital buffer high; reducing risk",
            extra={"buffer_pct": buffer_pct, "adjusted_risk": adjusted_risk},
        )
    elif buffer_pct > 0.10:
        adjusted_risk = 0.02 * 0.75
        logger.info(
            "Capital buffer elevated; adjusting risk",
            extra={"buffer_pct": buffer_pct, "adjusted_risk": adjusted_risk},
        )
    else:
        logger.info(
            "Capital buffer low; using full risk allocation",
            extra={"buffer_pct": buffer_pct},
        )

    last_daily_run = None
    last_audit_run = None

    portfolio_state = load_portfolio_state(refresh=True)

    # Kick off at least one suggestion write so dashboards have data on first run
    try:
        wrote_boot = run_learning_machine()
        if wrote_boot:
            logger.info("Boot suggestions written", extra={"count": wrote_boot})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": "scheduler.run_scheduler",
            "action": "run_learning_machine_initial",
            "message": "Initial learning machine execution failed",
            "error": str(exc),
        }
        anomalies_logger.info(json.dumps(error_payload, separators=(",", ":")))
        logger.error("Initial run_learning_machine failed", extra={"error": str(exc)})

    while True:
        try:
            if is_live:
                require_live_confirmation()

            # Refresh portfolio state and run trade evaluation
            portfolio_state = load_portfolio_state(refresh=True)
            available_capital = float(portfolio_state.get("available_capital", 0.0))
            reinvestment_rate = float(portfolio_state.get("reinvestment_rate", 0.0))

            if available_capital <= 0:
                logger.warning("Available capital is non-positive — skipping trade evaluation.")
            else:
                logger.info("Evaluating trades")
                evaluate_signals_and_trade(
                    tradable_pairs=CONFIG.get("tradable_pairs", []),
                    available_capital=available_capital,
                    risk_per_trade=adjusted_risk,
                    reinvestment_rate=reinvestment_rate,
                )

            logger.info("Checking exit conditions")
            run_exit_checks()

            # Run sync validation each cycle after exits
            try:
                validator = SyncValidator()
                ok = validator.validate_sync()
                if not ok:
                    logger.warning(
                        "Sync validation issues detected",
                        extra={"errors": list(validator.validation_errors)},
                    )
                else:
                    logger.info("Sync validation passed")
            except (ValueError, RuntimeError, OSError) as e:
                logger.error("SyncValidator failed", extra={"error": str(e)})

            # Run anomaly audit every 6 hours
            if should_run_anomaly_audit(last_audit_run):
                logger.info("Running anomaly audit")
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

            logger.info("Running scheduled tasks...")
            schedule.run_pending()

            time.sleep(TRADE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            break
        except ConfigurationError as error:
            try:
                anomalies_logger.critical(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "module": "scheduler",
                            "action": "require_live_confirmation",
                            "message": str(error),
                        },
                        separators=(",", ":"),
                    )
                )
            except (TypeError, ValueError, OSError):  # pragma: no cover
                pass
            logger.error("Scheduler configuration error", extra={"error": str(error)})
            raise
        except ValueError as error:
            logger.error("Scheduler value error", extra={"error": str(error)})
            traceback.print_exc()
        except RuntimeError as error:
            logger.error("Scheduler runtime error", extra={"error": str(error)})
            traceback.print_exc()
        except OSError as error:
            logger.error("Scheduler OS error", extra={"error": str(error)})
            traceback.print_exc()


if __name__ == "__main__":
    run_scheduler()
