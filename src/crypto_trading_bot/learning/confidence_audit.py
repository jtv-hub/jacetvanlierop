"""
confidence_audit.py

Audits the trade log to identify malformed or out-of-range confidence values.
"""

import fcntl
import json
import logging
import os
from datetime import datetime, timezone

from crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler
from crypto_trading_bot.scripts.sync_validator import SyncValidator

ANOMALY_LOG_PATH = "logs/anomalies.log"

logger = logging.getLogger("confidence_audit")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    rotating_handler = get_rotating_handler("anomalies.log")
    logger.addHandler(rotating_handler)


def log_anomaly(anomaly, source="audit"):
    """Log a malformed trade anomaly with timestamp and source."""
    os.makedirs("logs", exist_ok=True)
    anomaly_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        **anomaly,
    }
    with open(ANOMALY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(anomaly_record) + "\n")
        try:
            f.flush()
            os.fsync(f.fileno())
        except (OSError, IOError):
            pass


def load_trades(path):
    """Load all trades from a JSON-formatted log file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, IOError) as e:
        print(f"[Audit] Error loading trades from {path}: {e}")
        return []


def is_valid_confidence(value):
    """Check if a confidence value is a valid float between 0.0 and 1.0."""
    try:
        confidence = float(value)
        return 0.0 <= confidence <= 1.0
    except (ValueError, TypeError):
        return False


def is_valid_strategy(value):
    """Check if a strategy name is a non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def is_valid_size(value):
    """Check if a trade size is a positive number."""
    try:
        size = float(value)
        return size > 0
    except (ValueError, TypeError):
        return False


def audit_trades(trade_log_path, positions_file: str = "logs/positions.jsonl"):
    """Audit all trades to find entries with malformed data and lifecycle mismatches.

    The optional `positions_file` is used for cross-checking lifecycle consistency.
    """
    trades = load_trades(trade_log_path)
    bad_trades = []

    for i, trade in enumerate(trades):
        if not is_valid_confidence(trade.get("confidence")):
            anomaly = {
                "index": i,
                "type": "invalid_confidence",
                "value": trade.get("confidence"),
                "pair": trade.get("pair"),
                "strategy": trade.get("strategy"),
                "timestamp": trade.get("timestamp"),
                "trade_id": trade.get("trade_id"),
            }
            log_anomaly(anomaly, source="audit")
            bad_trades.append(anomaly)

        if not is_valid_strategy(trade.get("strategy")):
            anomaly = {
                "index": i,
                "type": "invalid_strategy",
                "value": trade.get("strategy"),
                "pair": trade.get("pair"),
                "confidence": trade.get("confidence"),
                "timestamp": trade.get("timestamp"),
                "trade_id": trade.get("trade_id"),
            }
            log_anomaly(anomaly, source="audit")
            bad_trades.append(anomaly)

        if not is_valid_size(trade.get("size")):
            anomaly = {
                "index": i,
                "type": "invalid_size",
                "value": trade.get("size"),
                "pair": trade.get("pair"),
                "strategy": trade.get("strategy"),
                "timestamp": trade.get("timestamp"),
                "trade_id": trade.get("trade_id"),
            }
            log_anomaly(anomaly, source="audit")
            bad_trades.append(anomaly)

        # Enhanced anomaly: status/exit_price mismatches
        status = trade.get("status")
        exit_price = trade.get("exit_price")
        if status == "closed" and exit_price is None:
            anomaly = {
                "index": i,
                "type": "closed_without_exit_price",
                "pair": trade.get("pair"),
                "strategy": trade.get("strategy"),
                "timestamp": trade.get("timestamp"),
                "trade_id": trade.get("trade_id"),
            }
            log_anomaly(anomaly, source="audit")
            bad_trades.append(anomaly)
        if status != "closed" and exit_price is not None:
            anomaly = {
                "index": i,
                "type": "open_with_exit_price",
                "value": exit_price,
                "pair": trade.get("pair"),
                "strategy": trade.get("strategy"),
                "timestamp": trade.get("timestamp"),
                "trade_id": trade.get("trade_id"),
            }
            log_anomaly(anomaly, source="audit")
            bad_trades.append(anomaly)

        # Check RSI field if present (should be within 0-100)
        rsi_val = trade.get("rsi")
        if rsi_val is not None:
            try:
                r = float(rsi_val)
                if r < 0.0 or r > 100.0:
                    anomaly = {
                        "index": i,
                        "type": "rsi_out_of_range",
                        "value": rsi_val,
                        "trade_id": trade.get("trade_id"),
                        "pair": trade.get("pair"),
                        "strategy": trade.get("strategy"),
                    }
                    log_anomaly(anomaly, source="audit")
                    bad_trades.append(anomaly)
            except (ValueError, TypeError):
                anomaly = {
                    "index": i,
                    "type": "rsi_invalid",
                    "value": rsi_val,
                    "trade_id": trade.get("trade_id"),
                }
                log_anomaly(anomaly, source="audit")
                bad_trades.append(anomaly)

        # Confidence must be within [0,1]
        conf_val = trade.get("confidence")
        if conf_val is not None:
            try:
                c = float(conf_val)
                if c < 0.0 or c > 1.0:
                    anomaly = {
                        "index": i,
                        "type": "confidence_out_of_range",
                        "value": conf_val,
                        "trade_id": trade.get("trade_id"),
                        "pair": trade.get("pair"),
                        "strategy": trade.get("strategy"),
                    }
                    log_anomaly(anomaly, source="audit")
                    bad_trades.append(anomaly)
            except (ValueError, TypeError):
                anomaly = {
                    "index": i,
                    "type": "confidence_invalid",
                    "value": conf_val,
                    "trade_id": trade.get("trade_id"),
                }
                log_anomaly(anomaly, source="audit")
                bad_trades.append(anomaly)

    # Cross-check for trades marked closed but still present in positions.jsonl
    try:
        validator = SyncValidator(
            trades_file=trade_log_path,
            positions_file=positions_file,
        )
        # Reuse its loaders with the provided paths
        trades_list = validator.load_json_lines(trade_log_path, "trades.log")
        positions = validator.load_json_lines(positions_file, "positions.jsonl")
        pos_ids = {p.get("trade_id") for p in positions if p.get("trade_id")}
        for i, t in enumerate(trades_list):
            if t.get("status") == "closed" and t.get("exit_price") is not None and t.get("trade_id") in pos_ids:
                anomaly = {
                    "index": i,
                    "type": "closed_trade_still_in_positions",
                    "trade_id": t.get("trade_id"),
                    "pair": t.get("pair"),
                    "strategy": t.get("strategy"),
                }
                log_anomaly(anomaly, source="audit")
                bad_trades.append(anomaly)
    except (OSError, IOError, ValueError, TypeError, json.JSONDecodeError) as e:
        # Do not break audit on validator failure
        log_anomaly({"type": "audit_validator_error", "error": str(e)}, source="audit")

    return bad_trades


def print_audit_report(results):
    """Print a formatted report of all trades with validation issues."""
    if not results:
        print("✅ All trades passed audit.")
        return

    print("⚠️ Audit issues found:")
    for r in results:
        print(
            f"- Index {r.get('index', 'N/A')} [{r.get('type', 'N/A')}]: "
            f"value={r.get('value', 'N/A')}, strategy={r.get('strategy')}, "
            f"pair={r.get('pair')}, time={r.get('timestamp')}"
        )


def cleanup_closed_positions(trades_file="logs/trades.log", positions_file="logs/positions.jsonl") -> int:
    """Remove any positions whose matching trade is closed from positions.jsonl.

    Returns the number of positions removed.
    """
    try:
        validator = SyncValidator()
        trades = validator.load_json_lines(trades_file, "trades.log")
        closed_ids = {
            t.get("trade_id") for t in trades if t.get("status") == "closed" and t.get("exit_price") is not None
        }
        if not os.path.exists(positions_file) or not closed_ids:
            return 0
        removed = 0
        tmp_path = positions_file + ".tmp"
        with (
            open(positions_file, "r", encoding="utf-8") as src,
            open(tmp_path, "w", encoding="utf-8") as dst,
        ):
            fcntl.flock(dst, fcntl.LOCK_EX)
            try:
                for line in src:
                    try:
                        obj = json.loads(line)
                        if obj.get("trade_id") in closed_ids:
                            removed += 1
                            continue
                    except json.JSONDecodeError:
                        pass
                    dst.write(line)
                dst.flush()
                os.fsync(dst.fileno())
            finally:
                fcntl.flock(dst, fcntl.LOCK_UN)
        os.replace(tmp_path, positions_file)
        return removed
    except (OSError, IOError, ValueError, TypeError, json.JSONDecodeError) as e:
        log_anomaly({"type": "positions_cleanup_error", "error": str(e)}, source="audit")
        return 0


def run_and_cleanup(trade_log_path="logs/trades.log", positions_file="logs/positions.jsonl") -> dict:
    """Run audit, perform cleanup of closed positions, rerun audit, and return summary.

    Returns a dict: {"initial_errors": int, "removed": int, "final_errors": int, "errors": [..]}
    """
    report1 = audit_trades(trade_log_path, positions_file=positions_file)
    removed = cleanup_closed_positions(trades_file=trade_log_path, positions_file=positions_file)
    report2 = audit_trades(trade_log_path, positions_file=positions_file)
    return {
        "initial_errors": len(report1),
        "removed": removed,
        "final_errors": len(report2),
        "errors": report2,
    }


def main(trade_log_path="logs/trades.log"):
    """Entry point for running the confidence audit externally.

    Also writes per-trade validation results to logs/validation_errors.log as JSONL.
    """
    report = audit_trades(trade_log_path)
    print_audit_report(report)

    # Summarize anomalies per trade and write JSONL results
    try:
        # Build anomalies by trade_id
        anomalies_by_trade = {}
        for a in report:
            tid = a.get("trade_id")
            if tid is None:
                continue
            anomalies_by_trade.setdefault(tid, []).append(a.get("type", "unknown"))

        # Load trades to get list of trade_ids
        trades = load_trades(trade_log_path)
        os.makedirs("logs", exist_ok=True)
        out_path = "logs/validation_errors.log"
        with open(out_path, "a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).isoformat()
            for t in trades:
                tid = t.get("trade_id")
                if not tid:
                    continue
                anomalies = anomalies_by_trade.get(tid, [])
                status = "fail" if anomalies else "pass"
                record = {
                    "timestamp": ts,
                    "trade_id": tid,
                    "anomalies": anomalies,
                    "status": status,
                }
                f.write(json.dumps(record) + "\n")
    except (OSError, IOError, ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"[Audit] Failed to write validation results: {e}")


if __name__ == "__main__":
    # Trigger a quick test anomaly to validate logging path
    try:
        log_anomaly(
            {"type": "test_anomaly", "message": "confidence_audit startup check"},
            source="selftest",
        )
    except (OSError, IOError, ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"[Audit] Failed to log test anomaly: {e}")
    main()
