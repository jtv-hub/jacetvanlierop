"""
Shadow-test deployment gate for PPO models.

Evaluates `logs/shadow_test_results.jsonl`, enforces strict safety metrics, and
records the approved PPO model in `models/approved_model.json`. The latest
confidence / PPO consumers can then load the approved model path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from crypto_trading_bot.utils.system_logger import get_system_logger

LOGGER = get_system_logger().getChild("deploy_gate")

SHADOW_RESULTS_PATH = Path("logs/shadow_test_results.jsonl")
APPROVED_MODEL_RECORD = Path("models/approved_model.json")
DEFAULT_MODEL_CANDIDATE = Path("models/ppo_agent_v1.zip")
MODEL_METADATA_PATH = Path("models/ppo_agent_v1_meta.json")

CRITERIA = {
    "min_trades": 100,
    "min_win_rate": 0.60,
    "min_sharpe": 1.30,
    "max_drawdown_pct": 0.10,
    "min_risk_adjusted_roi": 0.18,
    "max_model_age_hours": 168.0,  # 7 days
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
    return rows


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Unable to parse %s: %s", path, exc)
        return {}


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _latest_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for rec in reversed(rows):
        rec_type = rec.get("type")
        if rec_type == "ppo_shadow_summary":
            return rec
    return None


def _compute_risk_adjusted_roi(avg_roi: float, win_rate: float, max_drawdown_pct: float) -> float:
    if max_drawdown_pct <= 0:
        return 0.0
    return (avg_roi * win_rate) / max_drawdown_pct


def _model_age_hours(timestamp: str | None) -> float:
    if not timestamp:
        return float("inf")
    try:
        recorded = datetime.fromisoformat(timestamp)
    except ValueError:
        return float("inf")
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - recorded).total_seconds() / 3600.0


def check_and_approve(
    *,
    shadow_results_path: Path | str = SHADOW_RESULTS_PATH,
    candidate_model_path: Path | str = DEFAULT_MODEL_CANDIDATE,
    approved_record_path: Path | str = APPROVED_MODEL_RECORD,
    metadata_path: Path | str = MODEL_METADATA_PATH,
    model_path: str | None = None,
) -> Dict[str, Any]:
    """
    Evaluate latest shadow-test summary and decide whether to approve PPO model.

    Returns a dict with `approved`, `metrics`, and `reasons` keys for callers to log.
    """
    shadow_path = Path(shadow_results_path)
    rows = _load_jsonl(shadow_path)
    summary = _latest_summary(rows)

    if summary is None:
        LOGGER.warning("No shadow summary rows found in %s", shadow_path)
        result = {"approved": False, "reason": "missing_summary", "metrics": {}}
        LOGGER.info("Deploy gate result: %s", result)
        return result

    reasons: List[str] = []
    total_trades = _parse_int(summary.get("total_trades"), "total_trades", reasons)
    if total_trades is None:
        total_trades = 0
        reasons.append("total_trades_invalid")
    win_rate = _parse_float(summary.get("win_rate"), "win_rate", reasons)
    if win_rate is None:
        win_rate = 0.0
    avg_roi = _parse_float(summary.get("avg_roi"), "avg_roi", reasons)
    if avg_roi is None:
        avg_roi = 0.0
    sharpe = _parse_float(summary.get("sharpe"), "sharpe", reasons)
    if sharpe is None:
        sharpe = 0.0
    max_drawdown_value = _parse_float(summary.get("max_drawdown_pct"), "max_drawdown_pct", reasons)
    if max_drawdown_value is None:
        max_drawdown_value = 1.0
    summary_model_path = summary.get("model_path")
    model_ts = summary.get("model_timestamp")

    metadata = _safe_read_json(Path(metadata_path))
    meta_timestamp = metadata.get("timestamp")
    model_age_hours = _model_age_hours(model_ts or meta_timestamp)

    candidate_path = Path(summary_model_path or model_path or candidate_model_path)
    risk_adjusted_roi = _compute_risk_adjusted_roi(avg_roi, win_rate, max_drawdown_value)

    metrics = {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 6),
        "avg_roi": round(avg_roi, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown_pct": round(max_drawdown_value, 6),
        "risk_adjusted_roi": round(risk_adjusted_roi, 6),
        "model_age_hours": round(model_age_hours, 2),
    }

    if total_trades < CRITERIA["min_trades"]:
        reasons.append(f"insufficient_trades:{total_trades}")
    if sharpe < CRITERIA["min_sharpe"]:
        reasons.append(f"sharpe_below:{sharpe:.3f}")
    if max_drawdown_value > CRITERIA["max_drawdown_pct"]:
        reasons.append(f"drawdown_exceeds:{max_drawdown_value:.3f}")
    if risk_adjusted_roi < CRITERIA["min_risk_adjusted_roi"]:
        reasons.append(f"risk_adj_roi_below:{risk_adjusted_roi:.3f}")
    if model_age_hours > CRITERIA["max_model_age_hours"]:
        reasons.append(f"model_stale:{model_age_hours:.1f}h")
    if not candidate_path.exists():
        reasons.append(f"missing_model:{candidate_path}")

    LOGGER.info(
        "Shadow gate metrics: trades=%s win_rate=%.3f avg_roi=%.4f sharpe=%.3f "
        "drawdown=%.3f rar=%.3f age=%.1fh model=%s",
        total_trades,
        win_rate,
        avg_roi,
        sharpe,
        max_drawdown_value,
        risk_adjusted_roi,
        model_age_hours,
        candidate_path,
    )

    reason_text = "approved" if not reasons else "; ".join(sorted(set(reasons)))
    result = {
        "approved": not reasons,
        "reason": reason_text,
        "metrics": metrics,
    }

    if reasons:
        LOGGER.warning("PPO deployment gate rejected candidate: %s", "; ".join(reasons))
        LOGGER.info("Deploy gate result: %s", result)
        return result

    record_payload = {
        "model_path": str(candidate_path),
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "thresholds": CRITERIA,
        "metadata": metadata,
    }
    _atomic_write(Path(approved_record_path), record_payload)
    LOGGER.info("PPO deployment gate result: %s", result)
    return result


def get_latest_approved_model_path(default_model: str | Path = DEFAULT_MODEL_CANDIDATE) -> str:
    """Return path to the currently approved PPO model, or fallback to default."""
    record = _safe_read_json(APPROVED_MODEL_RECORD)
    approved_path = record.get("model_path")
    if isinstance(approved_path, str) and approved_path:
        return approved_path
    return str(default_model)


if __name__ == "__main__":
    check_and_approve()


def _parse_float(value: Any, name: str, reasons: List[str]) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        reasons.append(f"{name}_invalid")
        return None


def _parse_int(value: Any, name: str, reasons: List[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        reasons.append(f"{name}_invalid")
        return None
