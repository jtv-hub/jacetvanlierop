"""
Daily digest report.

Summarizes:
- Trades found in the learning ledger (today only)
- Confidence summary snapshot
- Last gatekeeper decision (if a proposals log exists)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

# Paths
TRADES_LOG = Path("logs/trades.log")  # optional; only used for a count
LEARNING_LOG = Path("logs/learning_ledger.jsonl")
CONF_SUMMARY = Path("logs/confidence_summary.json")
PROPOSALS_LOG = Path("logs/learning_proposals.jsonl")
REPORTS_DIR = Path("logs/reports")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file or return empty dict if missing/invalid."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file into a list of dicts, skipping invalid lines."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _today_learning(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter learning entries to today's date based on 'timestamp' prefix."""
    today = datetime.now().strftime("%Y-%m-%d")
    return [e for e in entries if str(e.get("timestamp", "")).startswith(today)]


def _summarize_learning(entries: List[Dict[str, Any]]) -> Tuple[int, float]:
    """Return count and average confidence for provided entries."""
    if not entries:
        return 0, 0.0
    confs = [float(e["confidence"]) for e in entries if e.get("confidence") is not None]
    avg = round(mean(confs), 4) if confs else 0.0
    return len(entries), avg


def _count_trades_lines(path: Path) -> int:
    """Count non-empty lines in trades.log (if present)."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def _latest_proposal() -> Dict[str, Any] | None:
    """Return the last JSON object from learning_proposals.jsonl, if present."""
    if not PROPOSALS_LOG.exists():
        return None
    last: Dict[str, Any] | None = None
    with PROPOSALS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def build_digest_text() -> str:
    """Build a human-readable daily digest string."""
    trades_count = _count_trades_lines(TRADES_LOG)
    learning = _read_jsonl(LEARNING_LOG)
    today_learning = _today_learning(learning)
    today_count, today_avg = _summarize_learning(today_learning)
    conf = _safe_read_json(CONF_SUMMARY)
    last_proposal = _latest_proposal()

    lines: List[str] = []
    lines.append("=== DAILY DIGEST ===")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Trades logged (all time): {trades_count}")
    lines.append(f"Learning entries today: {today_count} | Avg confidence: {today_avg:.4f}")
    lines.append("")

    if conf:
        lines.append("Confidence summary:")
        for strat, stats in conf.items():
            avgc = stats.get("average_confidence")
            cnt = stats.get("trade_count")
            lines.append(f"  - {strat}: avg={avgc}, trades={cnt}")
        lines.append("")
    else:
        lines.append("Confidence summary: (none)")
        lines.append("")

    if last_proposal:
        decision = last_proposal.get("decision", {})
        approved = "APPROVED" if decision.get("approved") else "REJECTED"
        lines.append("Last gatekeeper decision:")
        lines.append(f"  - Status: {approved}")
        metrics = decision.get("metrics", {})
        cov = metrics.get("coverage")
        dconf = metrics.get("delta_conf")
        orig = metrics.get("original_trades")
        shdw = metrics.get("shadow_trades")
        lines.append("  - Metrics: " f"orig={orig}, shadow={shdw}, coverage={cov}, Δconf={dconf}")
        reason = decision.get("reason")
        if reason:
            lines.append(f"  - Reason: {reason}")
    else:
        lines.append("Last gatekeeper decision: (none)")

    return "\n".join(lines)


def main() -> None:
    """Generate the digest, print it, and save under logs/reports/."""
    text = build_digest_text()
    print("\n" + text + "\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info("Saved digest: %s", out)


if __name__ == "__main__":
    main()
