#!/usr/bin/env python3
"""
Tiny SMTP sender used by notify.sh.

Environment variables
---------------------
Required:
  - SMTP_HOST
  - SMTP_PORT           (e.g. 587 for STARTTLS, 465 for SSL)
  - SMTP_USER
  - SMTP_PASS
  - EMAIL_TO

Optional:
  - EMAIL_FROM          (defaults to SMTP_USER or 'cryptobot@localhost')
  - SMTP_STARTTLS=1     (use STARTTLS when SMTP_SSL != 1)
  - SMTP_SSL=0          (set to 1 to use implicit SSL on connect)

Usage
-----
    email_notify.py "Subject" "Body"
"""

from __future__ import annotations

import os
import sys
import ssl
import smtplib
from email.message import EmailMessage


def main() -> int:
    """Parse CLI args, read SMTP config from env, and send the email."""
    subject = sys.argv[1] if len(sys.argv) > 1 else "Crypto Bot"
    body = sys.argv[2] if len(sys.argv) > 2 else "(no body)"

    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    to_addr = os.environ.get("EMAIL_TO")
    from_addr = os.environ.get("EMAIL_FROM", user or "cryptobot@localhost")

    starttls = os.environ.get("SMTP_STARTTLS", "1") != "0"
    use_ssl = os.environ.get("SMTP_SSL", "0") == "1"

    # Fail soft if config is incomplete (keeps notify.sh robust)
    if not all([host, user, password, to_addr]):
        print("email_notify: SMTP env not fully set; skipping.", file=sys.stderr)
        return 0

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(user, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as server:
            if starttls:
                context = ssl.create_default_context()
                server.starttls(context=context)
            server.login(user, password)
            server.send_message(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
