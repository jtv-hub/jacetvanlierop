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

LOG_FILE="${ROOT_DIR}/logs/system.log"
mkdir -p "${ROOT_DIR}/logs"

for var in KRAKEN_API_KEY KRAKEN_API_SECRET; do
    if [[ -z "${!var:-}" ]]; then
        echo "Environment variable ${var} is not set. Aborting." >&2
        exit 1
    fi
done

if [[ ! -f "${ROOT_DIR}/.confirm_live_trade" ]]; then
    echo "Confirmation sentinel .confirm_live_trade is missing. Create it before launching." >&2
    exit 1
fi

echo "[run_live] Running prelaunch checks..."
bash "${ROOT_DIR}/scripts/prelaunch_check.sh"

echo "[run_live] Running final health audit..."
bash "${ROOT_DIR}/scripts/final_health_audit.sh"

echo "[run_live] Launching live trading loop..."
cd "${ROOT_DIR}"
nohup "${PYTHON_BIN}" -m crypto_trading_bot.main --mode schedule --confirm-live-mode >>"${LOG_FILE}" 2>&1 &

PID=$!
echo "[run_live] Trading bot started with PID ${PID}. Logs -> ${LOG_FILE}"
echo "[run_live] Use scripts/kill_switch.sh pause to halt new trades if needed."
