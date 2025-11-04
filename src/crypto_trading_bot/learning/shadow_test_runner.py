"""
Shadow Test Runner
Runs shadow tests on strategy suggestions and logs results.
"""

import json
import logging
import os
import statistics
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

# === Setup rotating logger ===
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("shadow_test_runner")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "logs/shadow_test_runner.log",
    maxBytes=50 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# Canonical file paths
DEFAULT_INPUT = "logs/learning_feedback.jsonl"
DEFAULT_OUTPUT = "logs/shadow_test_results.jsonl"


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
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
            rows.append(rec)
    return rows


def run_shadow_tests(input_file: str = DEFAULT_INPUT, output_file: str = DEFAULT_OUTPUT) -> None:
    """Run shadow tests over learning suggestions in canonical JSONL format.

    Rules:
    - Only process entries where type=="learning_suggestion"
    - Pass criteria: median(confidence) >= 0.5, win_rate >= 0.55 (if present), ROI >= 0.003 (if present)
    - Append per-strategy summary rows
    """
    suggestions = _load_jsonl(input_file)
    suggestions = [s for s in suggestions if (s.get("type") == "learning_suggestion")]

    if not suggestions:
        logger.info("No learning_suggestion rows found in %s", input_file)
        return

    # Collect confidences for adaptive thresholding
    conf_values = []
    for s in suggestions:
        val = s.get("confidence_after") or s.get("suggested_confidence") or s.get("confidence")
        try:
            conf_values.append(float(val))
        except (TypeError, ValueError):
            continue

    threshold = 0.5
    if len(conf_values) > 1:
        try:
            threshold = max(0.5, float(statistics.median(conf_values)))
        except statistics.StatisticsError:
            threshold = 0.5

    results: list[dict] = []
    per_strategy: dict[str, dict[str, float]] = {}

    for s in suggestions:
        strategy = s.get("strategy") or s.get("strategy_name") or "Unknown"
        conf = s.get("confidence_after") or s.get("suggested_confidence") or s.get("confidence")
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0

        # Optional parameterized suggestion fields
        p_before = s.get("param_value_before")
        p_after = s.get("param_value_after")
        try:
            p_before = float(p_before) if p_before is not None else None
        except (TypeError, ValueError):
            p_before = None
        try:
            p_after = float(p_after) if p_after is not None else None
        except (TypeError, ValueError):
            p_after = None

        # Optional metrics (if provided by generator)
        avg_roi = s.get("avg_roi")
        win_rate = s.get("win_rate")
        sharpe = s.get("sharpe") or s.get("sharpe_ratio")
        try:
            avg_roi = float(avg_roi) if avg_roi is not None else None
        except (TypeError, ValueError):
            avg_roi = None
        try:
            win_rate = float(win_rate) if win_rate is not None else None
        except (TypeError, ValueError):
            win_rate = None
        try:
            sharpe = float(sharpe) if sharpe is not None else None
        except (TypeError, ValueError):
            sharpe = None

        # Apply gating
        passed = True
        reason_parts = []
        if conf < threshold:
            passed = False
            reason_parts.append(f"confidence {conf:.3f} < threshold {threshold:.3f}")
        if win_rate is not None and win_rate < 0.55:
            passed = False
            reason_parts.append(f"win_rate {win_rate:.3f} < 0.55")
        if avg_roi is not None and avg_roi < 0.003:
            passed = False
            reason_parts.append(f"avg_roi {avg_roi:.4f} < 0.003")
        reason = ", ".join(reason_parts) if reason_parts else "meets criteria"

        result = {
            "shadow_test_id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "strategy": strategy,
            "confidence": conf,
            "avg_roi": avg_roi,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "threshold": threshold,
            "status": "pass" if passed else "fail",
            "reason": reason,
            "success_rate": 1.0 if passed else 0.0,
        }
        results.append(result)

        stats = per_strategy.get(strategy) or {"passes": 0.0, "fails": 0.0, "conf_sum": 0.0}
        if passed:
            stats["passes"] += 1.0
        else:
            stats["fails"] += 1.0
        stats["conf_sum"] += float(conf)
        per_strategy[strategy] = stats

    # Append strategy summary
    if per_strategy:
        summary_rows = []
        for strat, st in per_strategy.items():
            total = st["passes"] + st["fails"]
            if total <= 0:
                continue
            summary_rows.append(
                {
                    "strategy": strat,
                    "tests": int(total),
                    "passes": int(st["passes"]),
                    "fails": int(st["fails"]),
                    "pass_rate": round(st["passes"] / total, 4),
                    "avg_confidence": round(st["conf_sum"] / total, 4),
                }
            )
        results.append(
            {
                "shadow_test_id": f"summary-{uuid.uuid4()}",
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "strategy_confidence_summary",
                "strategies": summary_rows,
            }
        )

    # Write JSONL with fsync safety
    os.makedirs(os.path.dirname(output_file) or "logs", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for rec in results:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
        out.flush()
        try:
            os.fsync(out.fileno())
        except OSError:
            pass
    logger.info("✅ Shadow test results saved to %s", output_file)


if __name__ == "__main__":
    run_shadow_tests()
