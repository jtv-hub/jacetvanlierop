"""
alert.py

Shared alert utility for logging alerts and optional webhook forwarding.
- Writes JSONL to logs/alerts.log
- If a webhook URL is provided, attempts to send and logs webhook failures with ERROR level
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

ALERTS_LOG_PATH = "logs/alerts.log"
MAX_ALERT_LOG_BYTES = int(os.getenv("CRYPTO_TRADING_BOT_ALERT_LOG_BYTES", str(5 * 1024 * 1024)))
ALERT_LOG_BACKUPS = int(os.getenv("CRYPTO_TRADING_BOT_ALERT_LOG_BACKUPS", "3"))


def _rotate_alert_log() -> None:
    """Rotate alerts.log when it grows beyond MAX_ALERT_LOG_BYTES."""

    try:
        if MAX_ALERT_LOG_BYTES <= 0 or not os.path.exists(ALERTS_LOG_PATH):
            return
        if os.path.getsize(ALERTS_LOG_PATH) < MAX_ALERT_LOG_BYTES:
            return
        for idx in range(ALERT_LOG_BACKUPS, 0, -1):
            if idx == 1:
                src = ALERTS_LOG_PATH
            else:
                src = f"{ALERTS_LOG_PATH}.{idx - 1}"
            dst = f"{ALERTS_LOG_PATH}.{idx}"
            if os.path.exists(src):
                os.replace(src, dst)
        # Start a fresh log file after rotation
        open(ALERTS_LOG_PATH, "w", encoding="utf-8").close()
    except OSError:
        # Best effort; ignore rotation failures in alert path
        pass


def _write_alert_line(payload: Dict[str, Any]) -> None:
    """Append a JSONL alert payload to logs/alerts.log (best-effort)."""
    try:
        os.makedirs("logs", exist_ok=True)
        with open(ALERTS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
            # Force persistence
            try:
                f.flush()
                os.fsync(f.fileno())
            except (OSError, IOError):
                # Best effort
                pass
        _rotate_alert_log()
    except (OSError, IOError):
        # Avoid raising in production alert path
        pass


def send_alert(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
    webhook_url: Optional[str] = None,
    timeout: int = 5,
) -> None:
    """
    Log an alert to logs/alerts.log as JSONL and optionally post to a webhook.

    Args:
        message: Human-readable message.
        context: Optional structured context payload.
        level: INFO | WARN | ERROR | CRITICAL
        webhook_url: Optional URL to POST the alert payload.
        timeout: seconds for webhook request.
    """
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
    }
    if context:
        payload["context"] = context

    # Always write to local alerts log
    _write_alert_line(payload)

    # Optional webhook
    if webhook_url:
        try:
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as _:
                pass
        except urllib.error.URLError as e:
            # Log webhook failure with ERROR level to alerts log
            failure_payload: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "message": "Alert webhook dispatch failed",
                "context": {"error": str(e), "original_message": message},
            }
            _write_alert_line(failure_payload)
