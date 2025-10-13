#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PAIR="${PAIR:-BTC/USDC}"
INTERVAL="${INTERVAL:-30}"      # seconds
MAX_SAMPLES="${MAX_SAMPLES:-60}"

sym_clean="$(echo "$PAIR" | tr -d '/ ' | tr '[:lower:]' '[:upper:]')"
out_dir="data/live"
mkdir -p "$out_dir"
out_file="${out_dir}/kraken_${sym_clean}_$(date +%Y%m%d).jsonl"

echo "[INFO] Capturing live Kraken data for ${PAIR}"
echo "[INFO] Interval: ${INTERVAL}s, Samples: ${MAX_SAMPLES}"
echo "📡 Recording ${PAIR} every $(printf '%.2f' "$INTERVAL")s → ${out_file} (JSONL)"

# --- Replace this block with your real capture if needed ---
# For now we just write heartbeat lines so playback has input.
i=0
while [ "$i" -lt "$MAX_SAMPLES" ]; do
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  # dummy price (stable so playback stays deterministic)
  printf '{"ts":"%s","pair":"%s","price":120000.10}\n' "$ts" "$PAIR" >> "$out_file"
  sleep "$INTERVAL"
  i=$((i+1))
done
# ----------------------------------------------------------

echo "✅ Done: wrote ${MAX_SAMPLES} samples."
echo "FILE: ${out_file}"
