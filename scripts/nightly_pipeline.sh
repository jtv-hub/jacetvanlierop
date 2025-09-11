#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"
LOG_DIR="$PROJECT_ROOT/logs"
STDOUT_LOG="$LOG_DIR/nightly_pipeline.out"
STDERR_LOG="$LOG_DIR/nightly_pipeline.err"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
SCHEDULER_SCRIPT="$PROJECT_ROOT/src/crypto_trading_bot/bot/scheduler.py"

mkdir -p "$LOG_DIR"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$SCHEDULER_SCRIPT" ]]; then
  echo "Scheduler script not found at $SCHEDULER_SCRIPT" >&2
  exit 1
fi

"$PYTHON_BIN" "$SCHEDULER_SCRIPT" >>"$STDOUT_LOG" 2>>"$STDERR_LOG"
