#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/jacevanlierop/crypto_trading_bot"
PIPELINE_SCRIPT="$PROJECT_ROOT/scripts/nightly_pipeline.sh"
SYSTEM_LOG="$PROJECT_ROOT/logs/system.log"
PIPELINE_LOG="$PROJECT_ROOT/logs/nightly_pipeline.out"
LAUNCH_AGENT_LABEL="com.jace.crypto.pipeline"
THRESHOLD_SECONDS=1800
NOW_SEC=$(date +%s)

log_msg() {
  local msg="$1"
  printf '[%s] WATCHDOG: %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$msg" | tee -a "$SYSTEM_LOG"
}

file_age_seconds() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "$THRESHOLD_SECONDS"
    return
  fi
  local mod
  mod=$(stat -f %m "$file" 2>/dev/null || stat -c %Y "$file" 2>/dev/null || echo "$NOW_SEC")
  echo $(( NOW_SEC - mod ))
}

is_pipeline_running() {
  pgrep -f "$PIPELINE_SCRIPT" >/dev/null 2>&1
}

needs_restart=true
if is_pipeline_running; then
  needs_restart=false
else
  age_system=$(file_age_seconds "$SYSTEM_LOG")
  age_pipeline=$(file_age_seconds "$PIPELINE_LOG")
  if (( age_system < THRESHOLD_SECONDS )) || (( age_pipeline < THRESHOLD_SECONDS )); then
    needs_restart=false
  fi
fi

if [[ "$needs_restart" == true ]]; then
  log_msg "pipeline inactive; restarting LaunchAgent $LAUNCH_AGENT_LABEL"
  launchctl kickstart -k "gui/$UID/$LAUNCH_AGENT_LABEL" || log_msg "launchctl kickstart failed"
else
  log_msg "pipeline healthy"
fi
