#!/usr/bin/env python3
"""
notify.py — tiny Slack webhook notifier.

Usage:
  python scripts/notify.py "Title line" "Optional long message"

Behavior:
- Reads the Slack webhook URL from the SLACK_WEBHOOK_URL env var.
- If SLACK_WEBHOOK_URL is missing, it prints a note and exits 0 (no crash).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Optional


def send_slack_message(title: str, message: Optional[str] = "") -> int:
    """Send a message to Slack via webhook.

    Args:
        title: Short, bolded first line in Slack.
        message: Optional details that follow on a new line.

    Returns:
        0 on success (or when webhook missing), 2 on usage error, 1 on HTTP error.
    """
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        # No webhook set — treat as a no-op so nightly jobs never fail because of Slack.
        print("SLACK_WEBHOOK_URL not set; skipping Slack notify.", file=sys.stderr)
        return 0

    payload = {"text": f"*{title}*\n{(message or '').strip()}".strip()}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Slack notify failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """CLI entrypoint: read args and send Slack message."""
    if len(sys.argv) < 2:
        print("Usage: notify.py <title> [message]", file=sys.stderr)
        return 2
    title = sys.argv[1]
    body = sys.argv[2] if len(sys.argv) > 2 else ""
    return send_slack_message(title, body)


if __name__ == "__main__":
    sys.exit(main())
