#!/usr/bin/env python
import json

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.safety.confirmation import require_live_confirmation

print("SAFETY CONFIRMATION")
print("=" * 50)
print(f"is_live: {CONFIG.get('is_live')}")
print(f"mode: {CONFIG.get('mode')}")
print(f"dry_run: {CONFIG.get('dry_run')}")
print(f"test_mode: {CONFIG.get('test_mode')}")

try:
    require_live_confirmation()
    print("LIVE CONFIRMATION: PASSED (unexpected!)")
except SystemExit:
    print("LIVE CONFIRMATION: BLOCKED (SAFE)")

print("\nFULL CONFIG:")
print(json.dumps(dict(CONFIG), indent=2))
