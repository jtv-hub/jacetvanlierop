"""Thread-safe JSONL logger for PPO training experiences."""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path("logs/ppo_experience.jsonl")
_log_lock = threading.Lock()


def log_experience(entry: Dict[str, Any]) -> None:
    """Log a single PPO experience entry as JSONL."""
    entry["timestamp"] = datetime.now(UTC).isoformat()
    line = json.dumps(entry, ensure_ascii=False)
    os.makedirs(LOG_PATH.parent, exist_ok=True)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
