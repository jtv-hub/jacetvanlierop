#!/usr/bin/env bash
# Nightly pipeline (cron friendly)
set -euo pipefail

# --------- env / paths ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_DEFAULT="$REPO_ROOT/venv/bin/python"
PY="${PY:-$PY_DEFAULT}"
export REPO_ROOT PY

echo "[env] Using PY=$PY"
"$PY" - <<'PYV'
import sys; print("Python", sys.version.split()[0])
PYV

# --------- (optional) fresh capture before playback ----------
if [[ -z "${SKIP_CAPTURE:-}" ]]; then
  if [[ -x "$REPO_ROOT/scripts/run_capture.sh" ]]; then
    echo "[capture] launching run_capture.sh"
    PAIRS="${PAIRS:-BTCUSD ETHUSD SOLUSD}" \
    CAPTURE_SECONDS="${CAPTURE_SECONDS:-45}" \
    CAPTURE_INTERVAL="${CAPTURE_INTERVAL:-5}" \
    "$REPO_ROOT/scripts/run_capture.sh" || true
  fi
fi

# --------- playback / review / summary ----------
# (Playback creates a recent live file and the review prints a short summary)
if [[ -x "$REPO_ROOT/playback.py" ]]; then
  echo "[playback] shim"
fi

echo "[review] auto_review"
"$PY" "$REPO_ROOT/auto_review.py" || true

echo "[nightly_email] building text body"
"$REPO_ROOT/scripts/nightly_email_summary.sh" || true

# --------- paper trades + learning ----------
echo "[paper] batch on latest live *.jsonl"
"$PY" "$REPO_ROOT/scripts/paper_trade.py" "$REPO_ROOT"/data/live/kraken_*_*.jsonl \
  --trail-mode pct --trail-pct 0.002 --trail-activate 0.000 --size 25 || true

echo "[learn] learn_from_paper"
"$PY" "$REPO_ROOT/scripts/learn_from_paper.py" || true

# --------- equity + dashboard ----------
MAT_OK=0
"$PY" -c "import matplotlib" >/dev/null 2>&1 && MAT_OK=1 || MAT_OK=0
if [[ "$MAT_OK" -eq 1 ]]; then
  echo "[equity] make_equity_from_paper"
  "$PY" "$REPO_ROOT/scripts/make_equity_from_paper.py" || true
else
  echo "[equity] SKIP: matplotlib not installed in venv ($PY)"
fi

echo "[dashboard] build_dashboard_html"
"$PY" "$REPO_ROOT/scripts/build_dashboard_html.py" || true

# Ensure the dashboard shows the latest equity image
if [[ -f "$REPO_ROOT/scripts/postprocess_dashboard_equity.py" ]]; then
  echo "[postprocess] inject equity_latest.png"
  "$PY" "$REPO_ROOT/scripts/postprocess_dashboard_equity.py" --verbose || true
fi

# --------- dev convenience (macOS) ----------
if command -v open >/dev/null 2>&1; then
  open "$REPO_ROOT/logs/reports/dashboard.html" || true
fi

# --------- email via Mail.app if RECIPIENT is set ----------
if command -v osascript >/dev/null 2>&1; then
  if [[ -n "${RECIPIENT:-}" && -x "$REPO_ROOT/scripts/send_nightly_email_macos.sh" ]]; then
    echo "[email] Sending nightly summary to $RECIPIENT via Mail.app"
    RECIPIENT="$RECIPIENT" \
    SUBJECT="Crypto Bot Nightly Summary — $(date '+%Y-%m-%d %H:%M %Z')" \
    "$REPO_ROOT/scripts/send_nightly_email_macos.sh" || true
  else
    echo "[email] RECIPIENT not set or helper missing; skipping email send"
  fi
fi

echo "[done] pipeline complete"
