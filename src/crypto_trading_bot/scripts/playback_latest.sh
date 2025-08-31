#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="./venv/bin/python"
LOG_DIR="logs"
OUT_LOG="$LOG_DIR/playback.out"
ERR_LOG="$LOG_DIR/playback.err"

mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] playback start…" >> "$OUT_LOG"

# Find newest live file
latest_file="$(ls -1t data/live/kraken_BTCUSD_*.jsonl 2>/dev/null | head -1 || true)"

if [[ -z "${latest_file}" || ! -f "${latest_file}" ]]; then
  echo "No live file found under data/live. Skipping." >> "$OUT_LOG"
  exit 0
fi

# (Optional safety) Only run if file is from last 24h
if [[ "$(find "${latest_file}" -mtime -1 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]]; then
  echo "Latest file is older than 24h (${latest_file}). Skipping." >> "$OUT_LOG"
  exit 0
fi

# Run playback with sensible defaults; tweak as you wish
"${PY}" scripts/playback_live_feed.py \
  --file "${latest_file}" \
  --size 100 \
  --rsi-threshold 40 \
  --tp 0.003 \
  --sl 0.003 \
  --max-hold 20 \
  --log-file logs/live_playback_trades.jsonl \
  --debug \
  >> "$OUT_LOG" 2>> "$ERR_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] playback done." >> "$OUT_LOG"
