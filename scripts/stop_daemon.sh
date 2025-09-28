#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_FILE="logs/bot_daemon.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found. Bot does not appear to be running."
  exit 0
fi

PID="$(cat "$PID_FILE" | tr -d '[:space:]')"
if [[ -z "$PID" ]]; then
  echo "PID file empty; removing stale file."
  rm -f "$PID_FILE"
  exit 0
fi

if ps -p "$PID" >/dev/null 2>&1; then
  kill "$PID"
  echo "Sent SIGTERM to bot process (PID $PID)."
else
  echo "No process with PID $PID found; removing stale PID file."
fi

rm -f "$PID_FILE"
