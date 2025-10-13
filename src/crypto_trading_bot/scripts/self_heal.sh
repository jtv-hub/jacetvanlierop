#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-$HOME/crypto_trading_bot}"
cd "$REPO"

log(){ printf '[%s] self-heal: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a logs/reports/health.log; }

# --- helpers ---------------------------------------------------------------

latest_live_file() {
  ls -t data/live/*BTCUSD*.jsonl 2>/dev/null | head -n1 || true
}

file_age_secs() {
  # prints age in seconds or "-" if missing
  local f="$1"
  if [[ ! -f "$f" ]]; then echo "-"; return 0; fi
  local now ts
  now=$(date +%s)
  ts=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo "$now")
  echo $(( now - ts ))
}

rotate_if_big() {
  local f="$1" max_bytes="${2:-1048576}" # 1 MiB
  if [[ -f "$f" ]]; then
    local sz; sz=$(wc -c < "$f" || echo 0)
    if (( sz > max_bytes )); then
      mv "$f" "${f}.bak.$(date +%Y%m%d_%H%M%S)" || true
      : > "$f"
      log "rotated large log: $(basename "$f")"
    fi
  fi
}

reload_agent() {
  local label="$1" plist="$2"
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load  "$plist"
  log "reloaded LaunchAgent $label"
}

kick_pipeline_once() {
  launchctl start com.jace.crypto.pipeline || true
  log "kicked pipeline via launchctl start"
}

# --- actions ---------------------------------------------------------------

log "starting self-heal…"

# 0) Make sure expected folders exist
mkdir -p logs/reports logs/snapshots data/live || true

# 1) Permissions: ensure scripts are executable
chmod +x scripts/*.sh auto_review.py || true

# 2) Rotate very large launchd logs (keeps tailing snappy)
rotate_if_big "logs/pipeline.launchd.out"
rotate_if_big "logs/pipeline.launchd.err"
rotate_if_big "logs/reports/auto_review.log"
rotate_if_big "logs/reports/health.log"

# 3) Try to reload the nightly pipeline LaunchAgent (schedule is already set)
PIPE_PLIST="$HOME/Library/LaunchAgents/com.jace.crypto.pipeline.plist"
if [[ -f "$PIPE_PLIST" ]]; then
  reload_agent "com.jace.crypto.pipeline" "$PIPE_PLIST"
  kick_pipeline_once
else
  log "pipeline plist missing; skipping reload"
fi

# 4) Fallback: run a quick playback if we at least have some data
if [[ -f scripts/nightly_pipeline.sh ]]; then
  # Try a quick “playback only” to update review artifacts fast
  MODE=playback ./scripts/nightly_pipeline.sh || true
  # If no data exists yet, do a tiny live capture to seed files
  if [[ -z "$(latest_live_file)" ]]; then
    PAIR="BTC/USDC" INTERVAL=5 MAX_SAMPLES=5 ./scripts/nightly_pipeline.sh || true
  fi
else
  log "nightly_pipeline.sh missing; cannot run fallback"
fi

# 5) Evaluate success
LIVE="$(latest_live_file || true)"
AGE="$(file_age_secs "${LIVE:-/dev/null}")"
CSV="logs/snapshots/metrics_snapshots.csv"
CSV_AGE="-"
if [[ -f "$CSV" ]]; then
  now=$(date +%s)
  ts=$(stat -f %m "$CSV" 2>/dev/null || stat -c %Y "$CSV" 2>/dev/null || echo "$now")
  CSV_AGE=$(( now - ts ))
fi

log "post-heal check: live_file=$(basename "${LIVE:-none}") live_age_s=${AGE:--} csv_age_s=${CSV_AGE:--}"

# success criteria: fresh live file (< 3600s) AND metrics csv updated (< 7200s)
if [[ "$AGE" != "-" && "$AGE" -lt 3600 && "$CSV_AGE" != "-" && "$CSV_AGE" -lt 7200 ]]; then
  log "self-heal result: OK"
  exit 0
fi

log "self-heal result: NOT_OK"
exit 2
