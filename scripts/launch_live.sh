#!/usr/bin/env bash
# Prelaunch helper for switching the crypto trading bot into live mode safely.
# NOTE: Ensures backups, dry-run validation, guard checks, then starts live scheduler.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export KRAKEN_API_KEY_FILE="$ROOT_DIR/.secrets/kraken_key.txt"
export KRAKEN_API_SECRET_FILE="$ROOT_DIR/.secrets/kraken_secret.txt"

if [[ ! -f "$KRAKEN_API_KEY_FILE" ]]; then
  echo "[WARN] Kraken API key file not found: $KRAKEN_API_KEY_FILE"
fi
if [[ ! -f "$KRAKEN_API_SECRET_FILE" ]]; then
  echo "[WARN] Kraken API secret file not found: $KRAKEN_API_SECRET_FILE"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
ALERT_CLEANER="$PYTHON_BIN scripts/clear_alerts.py --archive-if-over 5000"

timestamp="$(date -u +"%Y%m%d-%H%M%S")"
backup_dir="backups/prelaunch"
mkdir -p "$backup_dir"

echo "[1/6] Backing up logs to $backup_dir/prelaunch-${timestamp}.tar.gz"
tar -czf "$backup_dir/prelaunch-${timestamp}.tar.gz" logs 2>/dev/null || echo "⚠️ Log backup produced warnings"

echo "[2/6] Clearing oversized alerts log"
$ALERT_CLEANER

echo "[3/6] Clearing kill-switch flag if present"
rm -f logs/kill_switch.flag

echo "[4/6] Running one-off dry-run validation"
"$PYTHON_BIN" -m src.crypto_trading_bot.main --mode once --dry-run

echo "[5/6] Executing prelaunch guard checks"
"$PYTHON_BIN" -m src.crypto_trading_bot.safety.prelaunch_guard

echo "[6/6] Enabling live mode and starting scheduler"

exec "$PYTHON_BIN" -m src.crypto_trading_bot.main --mode schedule --confirm-live-mode
