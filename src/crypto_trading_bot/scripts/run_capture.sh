#!/usr/bin/env bash
# Capture multiple Kraken pairs in parallel.
# Env:
#   PAIRS="BTCUSD ETHUSD SOLUSD"
#   CAPTURE_SECONDS=60
#   CAPTURE_INTERVAL=5
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"

PAIRS="${PAIRS:-BTCUSD ETHUSD SOLUSD}"
CAPTURE_SECONDS="${CAPTURE_SECONDS:-60}"
CAPTURE_INTERVAL="${CAPTURE_INTERVAL:-5}"

echo "[capture] starting: pairs=[$PAIRS] seconds=$CAPTURE_SECONDS interval=$CAPTURE_INTERVAL"

pids=()
for pair in $PAIRS; do
  ( "$PY" "$REPO_ROOT/scripts/capture_kraken.py" \
      --pair "$pair" --seconds "$CAPTURE_SECONDS" --interval "$CAPTURE_INTERVAL" ) &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "[capture] one or more capture tasks failed (continuing)" >&2
fi
echo "[capture] done."
