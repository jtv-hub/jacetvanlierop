"""Audit tests ensuring recent NSGA-III JSONL artifacts are valid."""

from __future__ import annotations

import json
import os

import pytest

FILES = [
    "logs/nsga3_shadow_results.jsonl",
    "logs/nsga3_front.jsonl",
    "logs/nsga3_promotions.jsonl",
]


def _resolve_jsonl_path(base_path: str) -> str | None:
    """Return the first existing path between legacy/global filenames."""
    candidates = [base_path]
    if base_path.endswith(".jsonl"):
        candidates.insert(0, base_path.replace(".jsonl", "_global.jsonl"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@pytest.mark.parametrize("file_path", FILES)
def test_jsonl_files_exist_and_are_valid(file_path: str) -> None:
    """Each jsonl file should exist and contain JSON objects."""
    target = _resolve_jsonl_path(file_path)
    assert target is not None, f"{file_path} missing"
    with open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            assert isinstance(data, dict)
