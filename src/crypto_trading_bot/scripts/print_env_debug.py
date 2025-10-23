"""Print Kraken credential environment variables for debugging purposes."""

import os


def _mask(var: str, value: str | None) -> str:
    """Return a redacted view of sensitive environment variables."""
    if not value:
        return "[Not Set]"
    if var.endswith("_FILE"):
        return f"[File: {value}]"
    trimmed = value.strip()
    preview = trimmed[:4] if len(trimmed) >= 4 else "***"
    return f"{preview}*** (len={len(trimmed)})"


def main() -> None:
    """Display current environment configuration for Kraken credentials."""

    print("\n=== ENV DEBUG ===")
    for var in [
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
        "KRAKEN_API_KEY_FILE",
        "KRAKEN_API_SECRET_FILE",
    ]:
        print(f"{var}: {_mask(var, os.environ.get(var))}")


if __name__ == "__main__":
    main()
