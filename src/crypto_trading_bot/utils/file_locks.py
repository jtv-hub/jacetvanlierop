"""
Lightweight advisory file lock helpers.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from typing import Iterator, TextIO


@contextmanager
def advisory_lock(file_obj: TextIO) -> Iterator[None]:
    """Apply an exclusive advisory lock on an open file object."""
    fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock(file_path: str, mode: str = "a+") -> Iterator[TextIO]:
    """Backwards-compatible helper that opens and locks the given path."""
    with open(file_path, mode, encoding="utf-8") as handle:
        with advisory_lock(handle):
            yield handle
