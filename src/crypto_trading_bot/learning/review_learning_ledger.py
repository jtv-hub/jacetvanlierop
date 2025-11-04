"""
Review Learning Ledger
Analyzes trade logs and generates fallback learning suggestions.
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("review_learning_ledger")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


CANONICAL_OUT = "logs/learning_feedback.jsonl"


def review_ledger(loss_threshold: int = 3, window: int = 20) -> int:
    """Parse trades.log and emit fallback learning suggestions to canonical JSONL.

    - Only consider status=="closed" trades
    - Count losses per strategy over a rolling window
    - If losses >= threshold, emit a confidence reduction suggestion
    Returns number of suggestions written.
    """
    logger.info("[review] Starting ledger review...")

    ledger_file = "logs/trades.log"

    if not os.path.exists(ledger_file):
        logger.warning("[review] Ledger file not found: %s", ledger_file)
        return 0

    trades = []
    try:
        with open(ledger_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (t.get("status") or "").lower() != "closed":
                    continue
                trades.append(t)
    except OSError as e:
        logger.error("[review] Failed to read ledger: %s", e)
        return 0

    # Rolling window of most recent trades
    recent = trades[-window:]
    by_strategy: dict[str, list[dict]] = {}
    for t in recent:
        sname = t.get("strategy") or t.get("strategy_name") or "Unknown"
        by_strategy.setdefault(sname, []).append(t)

    suggestions: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    for strat, rows in by_strategy.items():
        losses = 0
        rois = []
        for r in rows:
            roi = r.get("roi")
            try:
                rv = float(roi)
                rois.append(rv)
                if rv < 0:
                    losses += 1
            except (TypeError, ValueError):
                continue
        if losses >= loss_threshold:
            # Heuristic: decrease confidence by 10% but clamp to [0.1, 1.0]
            conf_before = 0.6
            try:
                conf_before = float(rows[-1].get("confidence"))
            except (TypeError, ValueError):
                pass
            suggested = max(0.1, min(1.0, conf_before * 0.9))
            rationale = (
                (f"{losses} losses in last {len(rows)} closed trades; " f"avg_roi={sum(rois)/len(rois):.4f}")
                if rois
                else f"{losses} losses in last {len(rows)} closed trades"
            )
            suggestions.append(
                {
                    "timestamp": ts,
                    "type": "learning_suggestion",
                    "strategy": strat,
                    "confidence_before": conf_before,
                    "confidence_after": suggested,
                    "reason": rationale,
                    "status": "pending",
                }
            )

    if not suggestions:
        logger.info("[review] No suggestions generated.")
        return 0

    os.makedirs(os.path.dirname(CANONICAL_OUT), exist_ok=True)
    wrote = 0
    try:
        with open(CANONICAL_OUT, "a", encoding="utf-8") as f:
            for s in suggestions:
                f.write(json.dumps(s, separators=(",", ":")) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
            wrote = len(suggestions)
        logger.info("[review] Wrote %d suggestions to %s", wrote, CANONICAL_OUT)
    except OSError as e:
        logger.error("[review] Failed to write suggestions: %s", e)
        wrote = 0

    return wrote


def run() -> None:
    """Entrypoint for scripts."""
    review_ledger()


if __name__ == "__main__":
    run()
