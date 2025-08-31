#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# keep latest 30 summaries/trade logs
for PAT in 'logs/paper/paper_summary_*.json' 'logs/paper/paper_trades_*.jsonl'; do
  ls -1t $PAT 2>/dev/null | awk 'NR>30' | xargs -I{} rm -f "{}" 2>/dev/null || true
done
