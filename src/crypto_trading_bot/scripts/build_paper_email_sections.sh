#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Symbols to include in the email section (newest summary per symbol)
SYMS=("BTCUSD" "ETHUSD")

mkdir -p logs/reports
OUT="logs/reports/paper_email_section.txt"
: > "$OUT"

for SYM in "${SYMS[@]}"; do
  # newest paper summary whose "file" mentions this symbol
  SUM=$(grep -l "\"file\".*_${SYM}_" logs/paper/paper_summary_*.json 2>/dev/null | sort -r | head -1 || true)
  [ -z "${SUM:-}" ] && continue

  TRADES=$(grep -m1 '"trades"'         "$SUM" | sed -E 's/.*: *([0-9.]+).*/\1/')
  WR_RAW=$(grep -m1 '"win_rate"'       "$SUM" | sed -E 's/.*: *([0-9.]+).*/\1/')
  RET_RAW=$(grep -m1 '"ret_pct"'       "$SUM" | sed -E 's/.*: *([0-9.-]+).*/\1/')
  KILL=$(   grep -m1 '"kill_triggered"' "$SUM" | sed -E 's/.*: *(true|false|"(0|1)").*/\1/' | tr -d '"\r\n')

  WR_FMT=$(awk -v x="${WR_RAW:-0}" 'BEGIN{printf("%.2f", x*100)}')
  RET_FMT=$(awk -v x="${RET_RAW:-0}" 'BEGIN{printf("%.2f", x)}')

  ALERT=""
  if awk "BEGIN{exit !(${RET_RAW:-0} <= -2.0)}"; then
    ALERT="ALERT: ❗ return ${RET_FMT}% ≤ -2%"
  fi
  if [ "${TRADES:-0}" -ge 5 ] && awk "BEGIN{exit !(${WR_RAW:-0} < 0.30)}"; then
    ALERT="${ALERT:+$ALERT | }ALERT: ❗ win_rate ${WR_FMT}% < 30%"
  fi

  {
    echo "## Paper Trading — ${SYM}"
    echo "trades=${TRADES:-?}  win_rate=${WR_FMT}%  ret_pct=${RET_FMT}%"
    echo "summary: $SUM"
    PNG=$(ls -1t logs/backtests/equity_*.png 2>/dev/null | head -1 || true)
    [ -n "${PNG:-}" ] && echo "equity_plot: $PNG"
    if [ "$KILL" = "true" ] || [ "$KILL" = "1" ]; then
      echo "ALERT: ❗ kill-switch hit"
    fi
    [ -n "$ALERT" ] && echo "$ALERT"
    echo ""
  } >> "$OUT"
done
