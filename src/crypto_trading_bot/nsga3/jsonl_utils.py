"""Shared helpers for durable JSONL file operations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List


def load_jsonl(
    path: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> List[dict[str, Any]]:
    """Load JSON objects from a newline-delimited file."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    entries: List[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    entries.append(parsed)
    except OSError as exc:
        if logger:
            logger.warning("JSONL read failed for %s: %s", file_path, exc)
        return []
    return entries
