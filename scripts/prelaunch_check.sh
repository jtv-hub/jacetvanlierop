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

echo "[check] Checking live confirmation gate..."
"${PYTHON_BIN}" - <<'PY'
import logging
import sys

from crypto_trading_bot import config as bot_config
from crypto_trading_bot.config import ConfigurationError
from crypto_trading_bot.safety.confirmation import require_live_confirmation
from crypto_trading_bot.safety.risk_guard import check_pause, load_state
from crypto_trading_bot.utils.kraken_client import describe_api_permissions

logging.basicConfig(level=logging.INFO)

if not bot_config.is_live:
    print("Live mode disabled — confirmation gate bypassed.")
    sys.exit(0)

try:
    require_live_confirmation()
except ConfigurationError as exc:
    print(f"[FAIL] {exc}")
    sys.exit(1)
else:
    print("[OK] Live confirmation sentinel present.")

print(f"[info] Kraken API permissions: {describe_api_permissions()}")

state = load_state()
paused, reason = check_pause(state)
if paused:
    print(f"[FAIL] Risk guard paused: {reason}")
    sys.exit(2)
print("[OK] Risk guard ready.")
PY

STATUS=$?
if [[ ${STATUS} -ne 0 ]]; then
    exit ${STATUS}
fi

set +e
echo "[diag] Running live diagnostics..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/debug_live_diagnostics.py"
DIAG_STATUS=$?
set -e
if [[ ${DIAG_STATUS} -ne 0 ]]; then
    echo "Diagnostics exited with status ${DIAG_STATUS}." >&2
fi

exit 0
