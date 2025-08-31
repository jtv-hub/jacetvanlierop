#!/usr/bin/env bash
set -Eeuo pipefail
# Usage: ./scripts/notify.sh INFO "Title" "Body"
# Channels:
#  • macOS Notification Center (best‑effort)
#  • Email via SMTP if SMTP_* env + EMAIL_TO are set
# De‑dupe: suppress identical alerts within 10 minutes.

LEVEL="${1:-INFO}"
TITLE="${2:-Crypto Bot}"
BODY="${3:-(no message)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/reports"
CACHE_DIR="$ROOT_DIR/logs/.alertcache"
mkdir -p "$LOG_DIR" "$CACHE_DIR"

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }

# ---- De‑dupe (10 min) ----
key="$(printf '%s|%s|%s' "$LEVEL" "$TITLE" "$BODY" | shasum | awk '{print $1}')"
now_s=$(date +%s)
ttl=$((10*60))
cache_file="$CACHE_DIR/$key"
if [[ -f "$cache_file" ]]; then
  last=$(cat "$cache_file" 2>/dev/null || echo 0)
  if [[ $(( now_s - last )) -lt $ttl ]]; then
    printf '[%s] notify: deduped level=%s title=%s\n' "$(stamp)" "$LEVEL" "$TITLE" >> "$LOG_DIR/health.log" || true
    exit 0
  fi
fi
echo "$now_s" > "$cache_file" 2>/dev/null || true

# ---- Log locally ----
printf '[%s] notify: level=%s title=%s body=%s\n' "$(stamp)" "$LEVEL" "$TITLE" "$BODY" >> "$LOG_DIR/health.log" || true

# ---- macOS banner ----
if command -v osascript >/dev/null 2>&1; then
  osascript -e "display notification \"$BODY\" with title \"$TITLE\"" >/dev/null 2>&1 || true
fi

# ---- Email (optional) ----
if [[ -n "${SMTP_HOST:-}" && -n "${SMTP_USER:-}" && -n "${SMTP_PASS:-}" && -n "${EMAIL_TO:-}" ]]; then
  python3 "$ROOT_DIR/scripts/email_notify.py" "$LEVEL — $TITLE" "$BODY" >/dev/null 2>&1 || true
fi
