#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found (checked python3, python)." >&2
    exit 1
fi

usage() {
    cat <<'USAGE'
Usage: scripts/kill_switch.sh <pause|resume|status> [reason]

Examples:
  scripts/kill_switch.sh pause "Manual review"
  scripts/kill_switch.sh resume
  scripts/kill_switch.sh status
USAGE
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

COMMAND="$1"
shift || true

case "${COMMAND}" in
    pause)
        REASON="${1:-Manual kill switch activated}"
        KILL_SWITCH_REASON="${REASON}" "${PYTHON_BIN}" - <<'PY'
import os
from crypto_trading_bot.safety import risk_guard

reason = os.getenv("KILL_SWITCH_REASON", "Manual kill switch activated")
risk_guard.activate_pause(reason, trigger="manual", context={"command": "pause"})
print(f"[risk_guard] Pause activated: {reason}")
PY
        ;;
    resume)
        "${PYTHON_BIN}" - <<'PY'
from crypto_trading_bot.safety import risk_guard

risk_guard.resume_trading(context={"command": "resume"})
print("[risk_guard] Pause cleared; trading may resume.")
PY
        ;;
    status)
        "${PYTHON_BIN}" - <<'PY'
import json
from crypto_trading_bot.safety import risk_guard

state = risk_guard.load_state(force_reload=True)
print(json.dumps(state, indent=2))
PY
        ;;
    *)
        usage
        exit 1
        ;;
esac
