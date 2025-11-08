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

from crypto_trading_bot.bot.state.portfolio_state import (
    load_portfolio_state,
    refresh_portfolio_state,
)
from crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from crypto_trading_bot.bot.utils.log_rotation import (
    get_anomalies_logger,
    get_rotating_handler,
)
from crypto_trading_bot.config import CONFIG, ConfigurationError, get_mode_label, is_live
from crypto_trading_bot.learning.confidence_audit import (
    run_and_cleanup as audit_run_and_cleanup,
)
from crypto_trading_bot.learning.learning_machine import run_learning_cycle, run_learning_machine
from crypto_trading_bot.learning.optimization import detect_outliers
from crypto_trading_bot.learning.shadow_test_runner import run_shadow_tests
from crypto_trading_bot.safety import risk_guard
from crypto_trading_bot.safety.confirmation import require_live_confirmation

# from crypto_trading_bot.scripts.check_exit_conditions import main as run_exit_checks
from crypto_trading_bot.scripts.daily_heartbeat import run_daily_tasks
from crypto_trading_bot.scripts.shadow_confidence_test import run_shadow_confidence_test
from crypto_trading_bot.scripts.suggest_top_configs import (
    export_suggestions,
    generate_parameter_suggestions,
)
from crypto_trading_bot.scripts.sync_validator import SyncValidator
from crypto_trading_bot.utils.system_logger import get_system_logger

try:
    from crypto_trading_bot.rl.ppo_agent import PPOAgent
except ImportError:  # pragma: no cover - optional dependency
    PPOAgent = None

# Constants for task intervals in seconds
TRADE_INTERVAL = 5 * 60  # Every 5 minutes
DAILY_TASK_HOUR = 0  # Midnight UTC
DAILY_TASK_MINUTE = 5  # Buffer to ensure market data is updated
ANOMALY_AUDIT_INTERVAL = 6 * 60 * 60  # 6 hours in seconds
PPO_SHADOW_INTERVAL = 5 * 60  # Every 5 minutes
ALERTS_LOG_PATH = "logs/alerts.log"
SHADOW_RESULTS_PATH = "logs/shadow_test_results.jsonl"
SHADOW_OBS_PATH = "logs/shadow_cycle_observations.jsonl"
LEDGER_STATE_PATH = Path("logs/ledger_state.json")

anomalies_logger = get_anomalies_logger()
logger = get_system_logger().getChild("scheduler")

PPO_AGENT = None
if PPOAgent:
    try:
        _singleton_agent = PPOAgent()
        _singleton_agent.load_model()
        PPO_AGENT = _singleton_agent
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - defensive init
        print(f"[PPO] Failed to load PPO model: {exc}")
        PPO_AGENT = None


def _json_safe(value):
    """Convert nested values into JSON-serializable scalars."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):  # pragma: no cover - best effort
            return str(value)
    return str(value)


def _ppo_is_approved() -> bool:
    """Best-effort PPO approval check without coupling tests to agent internals."""

    if not PPO_AGENT:
        return False
    try:
        return bool(getattr(PPO_AGENT, "is_approved", lambda: False)())
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        return False


def _load_ledger_state_snapshot() -> dict:
    """Read the latest ledger_state.json snapshot, returning an empty dict on error."""

    if not LEDGER_STATE_PATH.exists():
        return {}
    try:
        with LEDGER_STATE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


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
    if not CONFIG.get("is_live"):
        logger.info("Skipping anomaly audit — not in live mode.")
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
        with open(SHADOW_OBS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        logger.info(
            "Shadow observation appended",
            extra={"path": SHADOW_OBS_PATH, "win_rate": win_rate, "num_exits": num_exits},
        )
    except (OSError, IOError, ValueError) as e:
        send_alert("update_shadow_test_results failed", context={"error": str(e)})
        # Non-fatal
        return


def run_ppo_shadow_inference() -> None:
    """Log PPO agent actions in shadow mode for observability."""
    if not PPO_AGENT:
        print("[PPO] Shadow inference skipped – agent not approved.")
        return

    try:
        approved = bool(getattr(PPO_AGENT, "is_approved", lambda: False)())
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        approved = False

    if not approved:
        print("[PPO] Shadow inference skipped – agent not approved.")
        return

    tradable_pairs = CONFIG.get("tradable_pairs", [])
    if not tradable_pairs:
        return

    now = datetime.now(timezone.utc).isoformat()
    os.makedirs("logs", exist_ok=True)

    for pair in tradable_pairs:
        try:
            context = {
                "strategy_id": "scheduler_shadow",
                "pair": pair,
                "confidence": 1.0,
                "timestamp": now,
                "indicators": {},
                "portfolio": {},
            }
            action = PPO_AGENT.get_action(state=None, context=context) or {}
            try:
                confidence = float(action.get("agent_confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            model_path = getattr(PPO_AGENT, "model_path", "")
            row = {
                "timestamp": now,
                "pair": pair,
                "ppo_action": _json_safe(action),
                "ppo_confidence": confidence,
                "model_path": model_path,
                "mode": "shadow",
                "type": "ppo_shadow_tick",
            }
            with open(SHADOW_RESULTS_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            logger.info(
                "[PPO] Shadow inference recorded",
                extra={"pair": pair, "confidence": confidence},
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[PPO] Error during shadow inference for {pair}: {exc}")
            logger.warning(
                "PPO shadow inference failed",
                extra={"pair": pair, "error": str(exc)},
            )


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


def should_run_shadow_inference(last_run_time):
    """Determine whether to run PPO shadow inference based on interval."""
    if last_run_time is None:
        return True
    return (datetime.now(timezone.utc) - last_run_time).total_seconds() >= PPO_SHADOW_INTERVAL


def run_scheduler():
    """Runs the main scheduler loop that handles trade evaluation and daily bot maintenance."""
    logger.info("Scheduler started; running bot tasks")
    if not CONFIG.get("is_live") and not CONFIG.get("test_mode"):
        logger.warning("Live mode disabled in configuration — scheduler will not start.")
        return
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
    last_shadow_inference = None

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

    if CONFIG.get("is_live") and CONFIG.get("live", {}).get("CONFIRM_LIVE_TRADING", False):
        try:
            from crypto_trading_bot.utils.kraken_client import kraken_place_order

            tradable_pairs = CONFIG.get("tradable_pairs") or ["BTC/USDC"]
            pair = tradable_pairs[0]
            kraken_place_order(pair, "buy", 0.0001, 1.0, validate=True)
            logger.info("Live trading validation trade submitted", extra={"pair": pair})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.critical("Live trading validation failed: %s", exc)
            return

    while True:
        try:
            if not CONFIG.get("is_live") and not CONFIG.get("test_mode"):
                logger.info("Live mode disabled — scheduler idle.")
                time.sleep(TRADE_INTERVAL)
                continue

            if is_live:
                require_live_confirmation()

            # Refresh portfolio state and run trade evaluation
            portfolio_state = load_portfolio_state(refresh=True)
            ledger_state = _load_ledger_state_snapshot()
            logger.info(
                "[heartbeat]",
                extra={
                    "mode": get_mode_label(),
                    "is_live": is_live,
                    "ppo_approved": _ppo_is_approved(),
                    "drawdown": ledger_state.get("drawdown_pct", ledger_state.get("last_drawdown")),
                    "regime": portfolio_state.get("regime"),
                },
            )
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

            # Exit checks are handled inside evaluate_signals_and_trade(); avoid double-trigger
            # logger.info("Checking exit conditions")
            # run_exit_checks()

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
            if bool(CONFIG.get("auto_pause", {}).get("force_exit_on_severe_drawdown", False)):
                try:
                    risk_guard.trigger_panic_exit_if_needed()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("trigger_panic_exit_if_needed failed", extra={"error": str(e)})

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

            if should_run_shadow_inference(last_shadow_inference):
                run_ppo_shadow_inference()
                last_shadow_inference = datetime.now(timezone.utc)

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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for running the scheduler in paper or live mode."""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto trading bot scheduler")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Run the scheduler in paper or live configuration (default: paper).",
    )
    args = parser.parse_args(argv)
    requested_mode = args.mode

    if requested_mode == "paper":
        CONFIG["is_live"] = False
        CONFIG["test_mode"] = True
    else:
        CONFIG["is_live"] = True
        CONFIG["test_mode"] = False

    run_scheduler()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
