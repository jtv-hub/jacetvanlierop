"""
Test runner for the nightly pipeline.

This script ensures the nightly pipeline can be imported and executed
without relative import issues.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path for IDE + runtime consistency
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Import directly from scripts (since PROJECT_ROOT is crypto_trading_bot/)
    from scripts import nightly_pipeline
except ModuleNotFoundError as e:
    logging.error("[test] Import failed: %s", e)
    sys.exit(1)


# Ensure nightly_pipeline is referenced so it's not flagged as unused
if __name__ == "__main__":
    # Smoke test: ensure nightly_pipeline can run without errors
    try:
        nightly_pipeline.run_nightly_pipeline()
        logging.info("[test] Nightly pipeline executed successfully.")
    except (RuntimeError, ImportError, Exception) as e:
        # Catch common expected errors, log them, and exit
        logging.error("[test] Nightly pipeline execution failed: %s", e)
        sys.exit(1)
