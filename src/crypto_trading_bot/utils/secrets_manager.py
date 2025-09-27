"""Centralised helpers for retrieving secrets.

Supports environment variables and file-based fallbacks (``*_FILE``).
This intentionally avoids reading from project-local .env files so that
credentials can be sourced from dedicated secret stores or injected env vars.
"""

from __future__ import annotations

import os
from pathlib import Path


class SecretNotFound(RuntimeError):
    """Raised when a requested secret cannot be located."""


def _read_secret_file(path: str | os.PathLike[str]) -> str | None:
    file_path = Path(path)
    try:
        data = file_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return data or None


def get_secret(name: str, *, optional: bool = False) -> str:
    """Fetch the value for ``name``.

    Lookup order:
      1. Environment variable ``name``
      2. File pointed to by ``name + '_FILE'``

    Args:
        name: Secret key identifier (e.g. ``"KRAKEN_API_KEY"``).
        optional: When True, return an empty string instead of raising.

    Raises:
        SecretNotFound: If the secret is unavailable and ``optional`` is False.
    """

    direct = os.getenv(name)
    if direct:
        return direct.strip()

    file_hint = os.getenv(f"{name}_FILE")
    if file_hint:
        file_secret = _read_secret_file(file_hint)
        if file_secret:
            return file_secret.strip()

    if optional:
        return ""

    raise SecretNotFound(f"Secret '{name}' not found in environment or *_FILE variables.")


__all__ = ["SecretNotFound", "get_secret"]
