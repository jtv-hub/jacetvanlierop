#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-$HOME/crypto_trading_bot}"
cd "$REPO"

log(){ printf '[%s] health: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a logs/reports/health.log; }

latest_live_file() {
  ls -t data/live/*BTCUSD*.jsonl 2>/dev/null | head -n1 || true
}

file_age_secs() {
  local f="$1"
  if [[ ! -f "$f" ]]; then echo "-"; return 0; fi
  local now ts
  now=$(date +%s)
  ts=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo "$now")
  echo $(( now - ts ))
}

status_once() {
  local LIVE CSV AGE_LIVE AGE_CSV
  LIVE="$(latest_live_file || true)"
  CSV="logs/snapshots/metrics_snapshots.csv"

  AGE_LIVE="-"
  [[ -n "$LIVE" ]] && AGE_LIVE="$(file_age_secs "$LIVE")"
  AGE_CSV="-"
  [[ -f "$CSV" ]] && AGE_CSV="$(file_age_secs "$CSV")"

  local LIVE_OK="no" CSV_OK="no"
  [[ "$AGE_LIVE" != "-" && "$AGE_LIVE" -lt 3600 ]] && LIVE_OK="yes"
  [[ "$AGE_CSV"  != "-" && "$AGE_CSV"  -lt 7200 ]] && CSV_OK="yes"

  printf '%s|%s|%s|%s\n' "$LIVE_OK" "$AGE_LIVE" "$CSV_OK" "$AGE_CSV"
}

mkdir -p logs/reports || true

# First pass
IFS='|' read -r LIVE_OK AGE_LIVE CSV_OK AGE_CSV < <(status_once)

if [[ "$LIVE_OK" == "yes" && "$CSV_OK" == "yes" ]]; then
  log "status=OK live_ok=$LIVE_OK live_age_s=${AGE_LIVE} csv_ok=$CSV_OK"
  exit 0
fi

# Try self-heal once
log "status=WARN live_ok=$LIVE_OK live_age_s=${AGE_LIVE} csv_ok=$CSV_OK — attempting self-heal"
if ./scripts/self_heal.sh; then
  # Re-check
  IFS='|' read -r LIVE_OK AGE_LIVE CSV_OK AGE_CSV < <(status_once)
  if [[ "$LIVE_OK" == "yes" && "$CSV_OK" == "yes" ]]; then
    log "status=OK_AFTER_HEAL live_ok=$LIVE_OK live_age_s=${AGE_LIVE} csv_ok=$CSV_OK"
    ./scripts/notify.sh "INFO" "Crypto Bot Health" "Recovered automatically (live ok, csv ok)."
    exit 0
  fi
fi

# Escalation path
log "status=ALERT live_ok=$LIVE_OK live_age_s=${AGE_LIVE} csv_ok=$CSV_OK — escalation"
# include a short tail from pipeline and reviewer to help you triage quickly
PIPE_ERR="$(tail -n 20 logs/pipeline.launchd.err 2>/dev/null || true)"
REV_LOG="$(tail -n 10 logs/reports/auto_review.log 2>/dev/null || true)"
MSG="Health ALERT — live_ok=${LIVE_OK}, csv_ok=${CSV_OK}. Check logs."
./scripts/notify.sh "ALERT" "Crypto Bot Health" "$MSG"
# also drop a small triage bundle into the health log
{
  echo "--- pipeline.err (tail) ---"
  [[ -n "$PIPE_ERR" ]] && echo "$PIPE_ERR" || echo "(no pipeline.err yet)"
  echo "--- auto_review.log (tail) ---"
  [[ -n "$REV_LOG" ]] && echo "$REV_LOG" || echo "(no auto_review.log yet)"
} >> logs/reports/health.log

exit 2
