#!/usr/bin/env bash
# Send the nightly summary via macOS Mail using AppleScript.
# Usage:
#   RECIPIENT="you@example.com" ./scripts/send_nightly_email_macos.sh
# Optional:
#   SUBJECT="Custom subject"
#   DASH_HTML=/abs/path/to/dashboard.html

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BODY_FILE="$REPO_ROOT/logs/reports/nightly_email_body.txt"
DASH_HTML="${DASH_HTML:-$REPO_ROOT/logs/reports/dashboard.html}"
RECIPIENT="${RECIPIENT:-}"
SUBJECT="${SUBJECT:-Crypto Bot Nightly Summary — $(date '+%Y-%m-%d %H:%M %Z')}"

if [[ -z "$RECIPIENT" ]]; then
  echo "[email] RECIPIENT is required (export RECIPIENT=email@domain)" >&2
  exit 1
fi
if [[ ! -f "$BODY_FILE" ]]; then
  echo "[email] body not found: $BODY_FILE" >&2
  exit 1
fi

# AppleScript will read the file contents via `do shell script "cat <file>"`
# so we don't have to escape newlines/quotes/emoji.
osascript <<OSA
set theRecipient to "${RECIPIENT}"
set theSubject to "${SUBJECT}"
set bodyPath to POSIX path of "${BODY_FILE}"
set dashPath to POSIX path of "${DASH_HTML}"

-- read file EXACTLY as-is (UTF-8 safe through shell)
set bodyText to do shell script "cat " & quoted form of bodyPath
set linkLine to "🔗 View dashboard: file://" & dashPath
set theBody to bodyText & linefeed & linefeed & linkLine

tell application "Mail"
  set theMessage to make new outgoing message with properties {subject:theSubject, content:theBody, visible:true}
  tell theMessage
    make new to recipient at end of to recipients with properties {address:theRecipient}
  end tell
  send theMessage
end tell
OSA

echo "[email] Sent to $RECIPIENT"
