#!/usr/bin/env bash
# Nightly orchestration wrapper:
# - Runs auto_review.py (which calls ledger review, gatekeeper, pnl snapshot, reports)
# - Optionally runs cleanup_logs.py (dry-run controllable via DRY_RUN_CLEANUP=1)
# - Posts a Slack summary via scripts/notify.py (no-op if SLACK_WEBHOOK_URL unset)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUT_LOG="$ROOT_DIR/logs/launchd_nightly_review.out"
ERR_LOG="$ROOT_DIR/logs/launchd_nightly_review.err"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/logs/reports" "$ROOT_DIR/logs/snapshots"

# log helper (tee to OUT_LOG)
log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$OUT_LOG"
}

# On any error, write logs + Slack ❌ then exit 1
on_err() {
  {
    echo
    echo "==== ERROR (tail of logs) ===="
    echo "--- $OUT_LOG (last 60) ---"
    tail -n 60 "$OUT_LOG" || true
    echo
    echo "--- $ERR_LOG (last 60) ---"
    tail -n 60 "$ERR_LOG" || true
  } >>"$OUT_LOG"

  # Slack notify (best-effort)
  python3 "$ROOT_DIR/scripts/notify.py" \
    "Nightly review ❌ failed" \
    "See logs.\n$(tail -n 15 "$ERR_LOG" 2>/dev/null || echo '(no stderr)')\n---\n$(tail -n 15 "$OUT_LOG" 2>/dev/null || echo '(no stdout)')" \
    >/dev/null 2>>"$ERR_LOG" || true

  exit 1
}
trap on_err ERR

# capture all stdout/stderr of the run into logs, but still show in console
exec > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)

log "starting nightly review…"
python -V

# 1) Run the main review (skip confidence bump in automation)
python "$ROOT_DIR/auto_review.py" --skip-confidence-bump

# 2) Optional cleanup (set DRY_RUN_CLEANUP=1 to preview, RUN_CLEANUP=0 to skip entirely)
RUN_CLEANUP="${RUN_CLEANUP:-1}"
if [[ "$RUN_CLEANUP" == "1" ]]; then
  if [[ "${DRY_RUN_CLEANUP:-0}" == "1" ]]; then
    log "Running cleanup_logs.py (dry-run)"
    python "$ROOT_DIR/scripts/cleanup_logs.py" --dry-run
  else
    log "Running cleanup_logs.py"
    python "$ROOT_DIR/scripts/cleanup_logs.py"
  fi
fi

# 3) Build Slack success summary (safe if files are missing)
LATEST_DIGEST="$(ls -1t "$ROOT_DIR"/logs/reports/digest_*.txt 2>/dev/null | head -n 1 || true)"
PLOT_LINE="$(tail -n 1 "$ROOT_DIR"/logs/snapshots/paper_pnl_snapshots.csv 2>/dev/null || true)"

TITLE="Nightly review ✅ completed"
BODY="Latest digest: ${LATEST_DIGEST##*/}\nSnapshot: ${PLOT_LINE:-'(no snapshot yet)'}"

python "$ROOT_DIR/scripts/notify.py" "$TITLE" "$BODY" >/dev/null 2>>"$ERR_LOG" || true

log "nightly review finished."