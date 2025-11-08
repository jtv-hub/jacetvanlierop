"""
PPO Shadow Logger

Logs divergence between strategy output and PPO agent output for shadow testing.
Used to audit whether PPO decisions outperform or underperform baseline logic.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

LOG_PATH = Path("logs/ppo_shadow_log.jsonl")
_lock = Lock()


def log_divergence(
    strategy_output: dict,
    ppo_output: dict,
    decision: dict,
    trade_id: str = None,
) -> None:
    """Logs a single shadow gate decision and output comparison to a JSONL file."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trade_id": trade_id,
        "use_ppo": decision.get("use_ppo", False),
        "reason": decision.get("reason", "unknown"),
        "strategy_confidence": strategy_output.get("confidence"),
        "ppo_confidence": ppo_output.get("confidence"),
        "strategy_action": strategy_output,
        "ppo_action": ppo_output,
    }

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except (OSError, TypeError, ValueError) as err:
        print(f"[ppo_shadow_logger] Logging failed: {err}")
