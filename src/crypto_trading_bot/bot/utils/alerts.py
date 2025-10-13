"""
alerts.py

High-level alert dispatch utilities supporting multi-channel notifications.
Integrates with the existing JSONL alert logger and can optionally forward
messages via email, Slack/webhook, or generic POST targets.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any, Dict, Iterable, Optional, Sequence

from .alert import send_alert as _log_alert

logger = logging.getLogger(__name__)

DEFAULT_SLACK_WEBHOOK = os.getenv("CRYPTO_TRADING_BOT_SLACK_WEBHOOK")
DEFAULT_ALERT_WEBHOOK = os.getenv("CRYPTO_TRADING_BOT_ALERT_WEBHOOK")


def _coerce_list(value: Optional[Sequence[str] | str]) -> list[str]:
    if value is None:
        env_default = os.getenv("CRYPTO_TRADING_BOT_ALERT_EMAIL_TO")
        if env_default:
            return [addr.strip() for addr in env_default.split(",") if addr.strip()]
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _send_email(
    subject: str,
    body: str,
    recipients: Optional[Sequence[str] | str] = None,
) -> None:
    targets = _coerce_list(recipients)
    if not targets:
        logger.debug("alerts:_send_email skipped (no recipients configured)")
        return

    host = os.getenv("CRYPTO_TRADING_BOT_ALERT_EMAIL_HOST")
    port = int(os.getenv("CRYPTO_TRADING_BOT_ALERT_EMAIL_PORT", "587"))
    username = os.getenv("CRYPTO_TRADING_BOT_ALERT_EMAIL_USER")
    password = os.getenv("CRYPTO_TRADING_BOT_ALERT_EMAIL_PASS")
    sender = os.getenv("CRYPTO_TRADING_BOT_ALERT_EMAIL_FROM", username or "alerts@example.com")

    if not host or not username or not password:
        logger.warning("alerts:_send_email missing SMTP credentials; skipping email dispatch.")
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(targets)
    message.set_content(body)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.starttls(context=context)
            smtp.login(username, password)
            smtp.send_message(message)
    except smtplib.SMTPException as exc:
        logger.error("alerts:_send_email SMTP error: %s", exc, exc_info=False)
    except OSError as exc:
        logger.error("alerts:_send_email transport error: %s", exc, exc_info=False)


def _post_webhook(url: str, payload: Dict[str, Any], *, timeout: int = 5) -> None:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout):
            return
    except urllib.error.URLError as exc:
        logger.error("alerts:_post_webhook failed for %s: %s", url, exc, exc_info=False)


def _send_slack(payload: Dict[str, Any], webhook: Optional[str] = None) -> None:
    target = webhook or DEFAULT_SLACK_WEBHOOK
    if not target:
        logger.debug("alerts:_send_slack skipped (no webhook configured)")
        return

    slack_payload = {
        "text": payload.get("message"),
        "attachments": [
            {
                "color": "#ff0000" if payload.get("level") in {"CRITICAL", "ERROR"} else "#439FE0",
                "fields": [
                    {"title": "Level", "value": payload.get("level", "INFO"), "short": True},
                    {
                        "title": "Timestamp",
                        "value": payload.get("timestamp"),
                        "short": True,
                    },
                ],
            }
        ],
    }
    context = payload.get("context")
    if isinstance(context, dict) and context:
        slack_payload["attachments"][0]["fields"].append(
            {
                "title": "Context",
                "value": json.dumps(context, indent=2, default=str),
                "short": False,
            }
        )
    _post_webhook(target, slack_payload)


def send_alert(
    message: str,
    *,
    level: str = "INFO",
    context: Optional[Dict[str, Any]] = None,
    email_recipients: Optional[Sequence[str] | str] = None,
    slack: bool | str = False,
    webhook_url: Optional[str] = None,
    extra_webhooks: Optional[Iterable[str]] = None,
) -> None:
    """
    Dispatch a structured alert across the configured channels.

    Args:
        message: Human-readable alert message.
        level: Severity level (INFO, WARNING, ERROR, CRITICAL).
        context: Optional structured metadata.
        email_recipients: Optional recipients for SMTP delivery. If omitted,
            defaults to CRYPTO_TRADING_BOT_ALERT_EMAIL_TO when available.
        slack: When truthy, dispatch to Slack. If set to a string, uses that webhook URL.
        webhook_url: Override for the base webhook POST target.
        extra_webhooks: Additional webhook URLs to invoke.
    """
    _log_alert(message, context=context, level=level, webhook_url=webhook_url or DEFAULT_ALERT_WEBHOOK)

    payload = {
        "message": message,
        "level": level,
        "context": context or {},
    }

    if email_recipients or os.getenv("CRYPTO_TRADING_BOT_ALERT_EMAIL_TO"):
        subject = f"[Trading Bot] {level}: {message}"
        body_lines = [
            message,
            "",
            "Context:",
            json.dumps(context or {}, indent=2, default=str),
        ]
        _send_email(subject, "\n".join(body_lines), recipients=email_recipients)

    if slack:
        webhook = DEFAULT_SLACK_WEBHOOK if isinstance(slack, bool) else slack
        _send_slack(payload, webhook=webhook)

    for extra in extra_webhooks or ():
        _post_webhook(extra, payload)


__all__ = ["send_alert"]
