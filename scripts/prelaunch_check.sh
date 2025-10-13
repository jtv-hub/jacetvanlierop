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

echo "[check] Checking live lock-in prerequisites..."
"${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

try:
    bot_config = importlib.import_module("crypto_trading_bot.config")
except Exception as exc:  # pragma: no cover - defensive import guard
    print(f"❌ Unable to load configuration module: {exc}")
    sys.exit(10)

ConfigurationError = bot_config.ConfigurationError
CONFIG = getattr(bot_config, "CONFIG", {})
IS_LIVE = getattr(bot_config, "IS_LIVE", False)
ROOT_PATH = Path(os.environ.get("ROOT_DIR", ".")).resolve()
SAFETY_DIR = ROOT_PATH / "src" / "crypto_trading_bot" / "safety"


def _load_safety_module(filename: str, attr: str):
    target_path = SAFETY_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_safety_{attr}", target_path)
    if not spec or not spec.loader:
        raise ImportError(f"Unable to load safety module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return getattr(module, attr)


require_live_confirmation = _load_safety_module("confirmation.py", "require_live_confirmation")
rg_spec = importlib.util.spec_from_file_location("_risk_guard", SAFETY_DIR / "risk_guard.py")
if not rg_spec or not rg_spec.loader:
    raise ImportError("Unable to load risk_guard module")
risk_guard = importlib.util.module_from_spec(rg_spec)
sys.modules[rg_spec.name] = risk_guard
rg_spec.loader.exec_module(risk_guard)  # type: ignore[arg-type]
describe_api_permissions = importlib.import_module(
    "crypto_trading_bot.utils.kraken_client"
).describe_api_permissions

logging.basicConfig(level=logging.INFO)

print(f"[info] IS_LIVE={IS_LIVE} runtime_is_live={bot_config.is_live}")
if CONFIG.get("live_mode"):
    live_cfg = CONFIG.get("live_mode", {})
else:
    live_cfg = bot_config.CONFIG.get("live_mode", {})
if live_cfg:
    if live_cfg.get("force_override"):
        print("⚠️ LIVE_FORCE override is active — confirmation sentinel enforcement bypassed.")

if not (IS_LIVE or bot_config.is_live):
    print("ℹ️ Live mode disabled — skipping confirmation and risk guard checks.")
    sys.exit(0)

try:
    require_live_confirmation()
except ConfigurationError as exc:
    print(f"❌ Confirmation check failed: {exc}")
    sys.exit(1)
else:
    print("✅ Live confirmation sentinel present.")

try:
    perms = describe_api_permissions()
    print(f"[info] Kraken API permissions: {perms}")
except Exception as exc:  # pragma: no cover - network dependent
    print(f"❌ Failed to query Kraken API permissions: {exc}")
    sys.exit(30)

state = risk_guard.load_state(force_reload=True)
paused, reason = risk_guard.check_pause(state)
if paused:
    print(f"❌ Risk guard paused: {reason}")
    sys.exit(2)
print("✅ Risk guard clear.")
summary = {
    "consecutive_failures": state.get("consecutive_failures"),
    "last_drawdown": state.get("last_drawdown"),
    "max_drawdown": state.get("max_drawdown"),
    "last_update": state.get("updated_at"),
}
print(f"[info] Risk guard snapshot: {summary}")
print("✅ Prelaunch safety checks passed.")
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

# Ensure diagnostics didn't leave the risk guard paused (best-effort).
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

root_path = Path(os.environ.get("ROOT_DIR", ".")).resolve()
spec = importlib.util.spec_from_file_location(
    "_risk_guard_post",
    root_path / "src" / "crypto_trading_bot" / "safety" / "risk_guard.py",
)
if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    refreshed = module.load_state(force_reload=True)
    paused, reason = module.check_pause(refreshed)
    if paused:
        print(f"[info] Auto-clearing risk guard pause after diagnostics: {reason}")
        module.resume_trading(context={"command": "prelaunch_check_auto_resume"})
PY

exit 0
