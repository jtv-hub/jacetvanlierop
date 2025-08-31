#!/usr/bin/env python3
"""
Sweep the project and fix absolute imports like
'from crypto_trading_bot.xyz' → 'from ..xyz'.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_PACKAGE = "crypto_trading_bot"

# regex to catch imports like: from crypto_trading_bot.scripts import xyz
PATTERN = re.compile(rf"from {TARGET_PACKAGE}(\.[\w\.]*) import (.+)")

def fix_file(path: Path):
    """
    Rewrites absolute imports of the form 'from crypto_trading_bot.xyz' into relative imports
    for the given file.
    """
    text = path.read_text(encoding="utf-8")

    def replacer(match):
        module_path, items = match.groups()
        # count how many levels deep this file is
        depth = len(path.relative_to(ROOT).parts) - 1
        rel_prefix = "." * depth if depth > 0 else "."
        return f"from {rel_prefix}{module_path} import {items}"

    new_text = PATTERN.sub(replacer, text)

    if new_text != text:
        print(f"[fix] {path}")
        path.write_text(new_text, encoding="utf-8")

def sweep():
    """
    Iterates through all Python files in the project and applies import fixes,
    skipping itself.
    """
    for py in ROOT.rglob("*.py"):
        if py.name == "fix_imports.py":
            continue
        fix_file(py)

if __name__ == "__main__":
    sweep()
    print("[done] Swept project for absolute imports.")
