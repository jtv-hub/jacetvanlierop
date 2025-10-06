"""Print Kraken credential environment variables for debugging purposes."""

import os


def main() -> None:
    """Display current environment configuration for Kraken credentials."""

    print("\n=== ENV DEBUG ===")
    for var in [
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
        "KRAKEN_API_KEY_FILE",
        "KRAKEN_API_SECRET_FILE",
    ]:
        value = os.environ.get(var)
        print(f"{var}: {value if value else '[Not Set]'}")


if __name__ == "__main__":
    main()
