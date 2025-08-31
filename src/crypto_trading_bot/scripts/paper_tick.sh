#!/usr/bin/env bash
# One-shot paper-trade tick: capture → size → trade (per symbol) → dashboard
set -euo pipefail

# --- Paths ---
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-${REPO}/venv/bin/python}"
[ -x "$PY" ] || PY=python3

LIVE_DIR="$REPO/data/live"
REPORT_DIR="$REPO/logs/reports"
STATE_DIR="$REPO/.state"
LOGFILE="$REPORT_DIR/paper_tick.log"

mkdir -p "$LIVE_DIR" "$REPORT_DIR" "$STATE_DIR"

log() {
  # tee to screen when interactive; always append to logfile
  local ts; ts="$(date '+[%Y-%m-%d %H:%M:%S]')"
  echo "$ts $*" | tee -a "$LOGFILE" >/dev/null
}

# --- Simple lock (works on macOS without flock) ---
LOCKDIR="$REPO/.paper_tick.lock"
if mkdir "$LOCKDIR" 2>/dev/null; then
  trap 'rm -rf "$LOCKDIR"' EXIT
else
  log "another paper_tick is running, exiting."
  exit 0
fi

# --- Config ---
SYMBOLS="${SYMBOLS:-BTCUSD ETHUSD SOLUSD}"

# --- Capture latest public prices (no API keys needed) ---
log "capture start (symbols: $SYMBOLS)"
# We keep capture warnings inside the log but never fail the tick for transient net errors
if ! "$PY" "$REPO/scripts/capture_kraken_public.py" $SYMBOLS >>"$LOGFILE" 2>&1; then
  log "[WARN] capture script returned non-zero (continuing)"
fi

# --- Position size (reinvestment schedule / bankroll sizing) ---
POS_SIZE="25.0"
if OUT="$("$PY" "$REPO/scripts/compute_paper_size.py" 2>/dev/null)"; then
  POS_SIZE="$(printf '%s' "$OUT" | tr -d '\r\n' | awk '{print $1+0}')"
fi
log "position size computed → $POS_SIZE"

# --- Per-symbol paper trade ---
log "tick start (symbols: $SYMBOLS)"

for S in $SYMBOLS; do
  # Find most recent live file for the symbol
  LATEST="$(ls -1t "$LIVE_DIR/kraken_${S}_"*.jsonl 2>/dev/null | head -1 || true)"
  if [ -z "$LATEST" ] || [ ! -f "$LATEST" ]; then
    log "$S: no live JSONL found."
    continue
  fi

  bytes="$(wc -c <"$LATEST" | tr -d ' ')"
  stamp_file="$STATE_DIR/${S}.size"

  if [ -f "$stamp_file" ]; then
    prev="$(cat "$stamp_file" 2>/dev/null || echo 0)"
  else
    prev=0
  fi

  if [ "${bytes:-0}" -gt "${prev:-0}" ]; then
    log "$S: NEW $(basename "$LATEST") $(printf '(%8d bytes)' "$bytes")"
    echo "$bytes" >"$stamp_file"
  else
    log "$S: using $(basename "$LATEST") $(printf '(%8d bytes)' "$bytes")"
  fi

  # Run the paper trade for this symbol
  if "$PY" "$REPO/scripts/paper_trade.py" "$S" "$LATEST" "$POS_SIZE" >>"$LOGFILE" 2>&1; then
    log "$S: processed OK."
  else
    log "[WARN] paper_trade($S) failed (see $LOGFILE)"
  fi
done

# --- Rebuild dashboard every tick so the browser stays fresh ---
if "$PY" "$REPO/scripts/build_dashboard_html.py" >>"$LOGFILE" 2>&1; then
  :
else
  log "[WARN] dashboard build failed (see $LOGFILE)"
fi

log "tick done"
