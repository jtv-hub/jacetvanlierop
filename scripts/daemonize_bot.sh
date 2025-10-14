#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_FILE="logs/bot_daemon.pid"
LOG_FILE="logs/daemon.out"

mkdir -p logs

rotate_log() {
  local file="$1"
  local max_bytes=$((10 * 1024 * 1024))
  local backups=3

  if [[ ! -f "$file" ]]; then
    return
  fi

  local current_size
  current_size=$(wc -c < "$file")
  if (( current_size < max_bytes )); then
    return
  fi

  if [[ -f "${file}.${backups}" ]]; then
    rm -f "${file}.${backups}"
  fi

  for ((i=backups-1; i>=1; i--)); do
    if [[ -f "${file}.${i}" ]]; then
      mv "${file}.${i}" "${file}.$((i+1))"
    fi
  done
  mv "$file" "${file}.1"
}

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" | tr -d '[:space:]')"
  if [[ -n "$existing_pid" ]] && ps -p "$existing_pid" >/dev/null 2>&1; then
    echo "Bot scheduler already running (PID $existing_pid)."
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

if [[ -f ".env" ]]; then
  # shellcheck disable=SC2046
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

rotate_log "$LOG_FILE"

nohup python -m src.crypto_trading_bot.main --mode schedule > "$LOG_FILE" 2>&1 &
DAEMON_PID=$!
echo "$DAEMON_PID" > "$PID_FILE"
echo "Bot launched in background (PID $DAEMON_PID). Log: $LOG_FILE"
