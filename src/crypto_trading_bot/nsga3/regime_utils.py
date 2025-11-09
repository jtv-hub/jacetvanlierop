"""Shared helpers for regime-aware NSGA-III workflows."""

from __future__ import annotations


def normalize_regime_label(candidate: str | None) -> str:
    """Return a lowercase, filesystem-safe regime label."""
    label = str(candidate or "global").strip().lower()
    if not label:
        return "global"
    sanitized = "".join(ch for ch in label if ch.isalnum() or ch in ("-", "_"))
    return sanitized or "global"
