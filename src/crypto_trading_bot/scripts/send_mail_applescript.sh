#!/usr/bin/env bash
set -euo pipefail

SUBJECT="${1:?subject missing}"
BODY_FILE="${2:?body file missing}"
shift 2 || true
ATTACH=( "$@" )

: "${EMAIL_TO:?Set EMAIL_TO in your shell (e.g., export EMAIL_TO='you@example.com')}"

# Pass subject, recipient, body path, and each attachment as argv to AppleScript.
# We don't escape the subject here; osascript argv carries Unicode (emoji) fine.
# AppleScript reads the body via `do shell script "cat <path>"`.
/usr/bin/osascript - "$SUBJECT" "$EMAIL_TO" "$BODY_FILE" "${ATTACH[@]:-}" <<'OSA'
on run argv
  set subj          to item 1 of argv
  set recipientAddr to item 2 of argv
  set bodyPath      to item 3 of argv
  set attachList    to {}
  if (count of argv) > 3 then set attachList to items 4 thru -1 of argv

  set bodyText to do shell script "cat " & quoted form of bodyPath

  tell application "Mail"
    activate
    set msg to make new outgoing message with properties {visible:false, subject:subj, content:bodyText}
    tell msg
      make new to recipient with properties {address:recipientAddr}
      repeat with p in attachList
        try
          make new attachment with properties {file name:(POSIX file p)} at after the last paragraph
        end try
      end repeat
    end tell
    send msg
  end tell
end run
OSA

echo "[mail] sent via AppleScript to $EMAIL_TO with ${#ATTACH[@]} attachment(s)"
