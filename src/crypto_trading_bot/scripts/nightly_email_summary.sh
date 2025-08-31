#!/usr/bin/env bash
set -euo pipefail
PY_BIN="${PY:-python3}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Always run from repo root
cd "$(dirname "$0")/.."

# ---------- dynamic subject from paper stats ----------
if [ -f logs/paper/paper_daily.csv ]; then
  LAST=$(tail -n 1 logs/paper/paper_daily.csv)
  RET=$(echo "$LAST" | awk -F, '{print $15+0}')   # ret_pct
  WR=$( echo "$LAST" | awk -F, '{print $12+0}')   # win_rate (0..1)
  WRP=$(awk -v x="$WR" 'BEGIN{printf("%.0f", x*100)}')
  BAD=$(awk -v r="$RET" 'BEGIN{print (r < 0 ? "❗" : "")}')
  export EMAIL_SUBJECT="${EMAIL_SUBJECT:-Crypto Bot} ${BAD} ret=$(printf "%.2f" "$RET")% wr=${WRP}%"
fi
# ---------- end dynamic subject ----------

# Build the multi‑symbol paper section just-in-time
PAPER_SECTION="logs/reports/paper_email_section.txt"
./scripts/build_paper_email_sections.sh || true

# Prepare a body file (most of our blocks append to it)
BODY_FILE="${BODY_FILE:-logs/reports/nightly_email_body.txt}"
mkdir -p "$(dirname "$BODY_FILE")"
: > "$BODY_FILE"

# Header
{
  echo "📊 Crypto Bot Nightly Summary"
  echo "Date: $(date)"
  echo ""
  echo "🧾 Latest Metrics:"
  echo "Position size \(current\): $("$PY_BIN" scripts/compute_paper_size.py)"
} >> "$BODY_FILE"

# Include the paper section if present
if [ -s "$PAPER_SECTION" ]; then
  {
    echo ""
    echo "----------------------------------------"
    echo "Paper Trading"
    echo "----------------------------------------"
    cat "$PAPER_SECTION"
  } >> "$BODY_FILE"
fi

# ---------- optional AppleScript mail with PNG attachment ----------
if [ "${MAIL_USE_APPLESCRIPT:-0}" = "1" ]; then
  # latest equity png (if interesting)
  LATEST_PNG="$(ls -1t logs/backtests/equity_*.png 2>/dev/null | head -1 || true)"
  ATTACH_ARGS=()

  # attach only if interesting: |ret| >= 1% OR win_rate <= 35%
  COND_OK=0
  if [ -f logs/paper/paper_daily.csv ]; then
    L=$(tail -n 1 logs/paper/paper_daily.csv)
    RET=$(echo "$L" | awk -F, '{print $15+0}')
    WR=$( echo "$L" | awk -F, '{print $12+0}')
    awk -v r="$RET" -v w="$WR" 'BEGIN{exit !((r<=-1)||(r>=1)||(w<=0.35))}' && COND_OK=1 || COND_OK=0
  fi
  [ "$COND_OK" = "1" ] && [ -n "$LATEST_PNG" ] && [ -f "$LATEST_PNG" ] && ATTACH_ARGS+=("$LATEST_PNG")

  SUBJECT="${EMAIL_SUBJECT:-Crypto Bot Nightly Summary}"

# --- dashboard link footer ---
: "${REPO_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${BODY_FILE:=${REPO_ROOT}/logs/reports/nightly_email_body.txt}"
DASH_HTML="$REPO_ROOT/logs/reports/dashboard.html"
DASH_URL="file://$DASH_HTML"
printf '
🔗 View dashboard: file://$REPO_ROOT/logs/reports/dashboard.html
' "$DASH_URL" >> "$BODY_FILE"
# --- end dashboard link ---


  ./scripts/send_mail_applescript.sh "$SUBJECT" "$BODY_FILE" "${ATTACH_ARGS[@]:-}" || true
  # Avoid duplicate sending by exiting here.
  exit 0
fi
# ---------- end optional AppleScript mail with PNG attachment ----------

# Default behavior (stdout). Your pipeline prints this and/or sends via its normal path.
cat "$BODY_FILE"
