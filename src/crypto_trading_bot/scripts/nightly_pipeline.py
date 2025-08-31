"""
Nightly Pipeline Orchestrator
Coordinates ingestion, gatekeeping, and learning into a single nightly run.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# === Imports from scripts and learning ===
from scripts import ingest_paper_trades, gatekeeper
from learning import learning_pipeline

# === Setup rotating logger ===
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("nightly_pipeline")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "logs/nightly_pipeline.log",
    maxBytes=50 * 1024 * 1024,  # 50 MB
    backupCount=3,
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


def run_pipeline():
    """
    Core nightly pipeline:
    1. Ingest paper trades
    2. Run gatekeeper checks
    3. Trigger learning pipeline
    """
    logger.info("🚀 Starting nightly pipeline")

    # Step 1: Ingest trades
    logger.info("Step 1: Ingesting paper trades...")
    ingest_paper_trades.run()

    # Step 2: Gatekeeper validation
    logger.info("Step 2: Running gatekeeper checks...")
    gatekeeper.run()

    # Step 3: Run learning pipeline
    logger.info("Step 3: Running learning pipeline...")
    learning_pipeline.run_learning_pipeline()

    logger.info("✅ Nightly pipeline completed successfully")


def run_nightly_pipeline():
    """
    Wrapper for consistency with learning_pipeline.run_learning_pipeline().
    Calls run_pipeline().
    """
    return run_pipeline()


if __name__ == "__main__":
    run_nightly_pipeline()
