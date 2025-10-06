#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIRM_FILE="${LIVE_CONFIRMATION_FILE:-.confirm_live_trade}"

if [[ "${CONFIRM_FILE}" == /* ]]; then
    CONFIRM_PATH="${CONFIRM_FILE}"
else
    CONFIRM_PATH="${ROOT_DIR}/${CONFIRM_FILE}"
fi

if [[ -f "${CONFIRM_PATH}" ]]; then
    rm "${CONFIRM_PATH}"
    echo "Live confirmation disabled at ${CONFIRM_PATH}"
else
    echo "No live confirmation file found at ${CONFIRM_PATH}; nothing to remove."
fi
