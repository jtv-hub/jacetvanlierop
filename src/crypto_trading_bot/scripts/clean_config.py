"""
Utility script to deduplicate ``min_confidence`` entries inside CONFIG.

Keeps the lowest ``min_confidence`` discovered anywhere in the configuration,
promotes it to the top level, and persists the cleaned config.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from crypto_trading_bot.config import CONFIG, save_config


def find_min_confidence(obj: Any, path: str = "") -> List[Tuple[str, float]]:
    """Return a list of (path, value) for all ``min_confidence`` occurrences."""
    values: List[Tuple[str, float]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}{key}"
            if key == "min_confidence" and isinstance(value, (int, float)):
                values.append((current_path, float(value)))
            values.extend(find_min_confidence(value, current_path + "."))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            values.extend(find_min_confidence(item, f"{path}[{index}]."))
    return values


def _load_config_data() -> Dict[str, Any]:
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as source:
                return json.load(source)
        except (OSError, json.JSONDecodeError):
            pass
    return dict(CONFIG)


def main(*, force: bool = False) -> None:
    config_path = Path("config.json")
    if config_path.exists():
        backup_path = config_path.with_suffix(".json.bak")
        shutil.copy(config_path, backup_path)

    config_data = _load_config_data()
    occurrences = find_min_confidence(config_data)
    if len(occurrences) <= 1 and not force:
        print("No duplicates found.")
        return

    if not occurrences:
        print("No min_confidence values detected.")
        return

    min_val = min(value for _, value in occurrences)

    def remove_key(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return {k: remove_key(v, key) for k, v in obj.items() if k != key}
        if isinstance(obj, list):
            return [remove_key(item, key) for item in obj]
        return obj

    config_clean = remove_key(dict(CONFIG), "min_confidence")
    config_clean["min_confidence"] = round(min_val, 10)

    CONFIG.clear()
    CONFIG.update(config_clean)
    save_config()
    print(f"Cleaned: kept min_confidence = {min_val} (NSGA-III)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate min_confidence values in CONFIG.")
    parser.add_argument("--force", action="store_true", help="Run even if no duplicates are found.")
    cli_args = parser.parse_args()
    main(force=cli_args.force)
