"""
Shared file locking utilities.

Provides a simple context manager that acquires an exclusive lock on the
target file using ``fcntl.flock`` while the file is open.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager


@contextmanager
def _locked_file(path: str, mode: str = "r"):
    """Open ``path`` and hold an exclusive lock for the duration of the context."""

    with open(path, mode, encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
