"""
log_rotation.py

Provides rotating log handlers and optional compression for the crypto trading bot.
"""

import gzip
import logging
import os
import shutil
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
# Standardize rotation policy: 10MB, keep 3 backups
MAX_LOG_SIZE = 10 * 1024 * 1024
BACKUP_COUNT = 3


def get_rotating_handler(log_file: str) -> RotatingFileHandler:
    """
    Sets up a rotating log handler for the given log file.
    Automatically compresses older backups beyond the last one.

    Args:
        log_file (str): The filename (e.g., 'trades.log')

    Returns:
        RotatingFileHandler: Configured rotating handler
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_file)

    handler = RotatingFileHandler(
        filename=log_path,
        mode="a",
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )

    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.rotator = compress_old_log  # Called automatically after rollover
    handler.namer = compress_namer

    return handler


_ANOMALIES_LOGGER_NAME = "anomalies_logger"


def get_anomalies_logger() -> logging.Logger:
    """Return a shared rotating logger for logs/anomalies.log.

    - Rotates at MAX_LOG_SIZE with BACKUP_COUNT
    - UTF-8 encoding; compact message-only lines (JSONL provided by caller)
    - Singleton per process to avoid duplicate handlers
    """
    logger = logging.getLogger(_ANOMALIES_LOGGER_NAME)
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "anomalies.log"),
        mode="a",
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def compress_old_log(source: str, dest: str):
    """
    Compresses a rotated log file to .gz format.

    Args:
        source (str): Source file path
        dest (str): Destination file path
    """
    with open(source, "rb") as f_in, gzip.open(dest + ".gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(source)


def compress_namer(default_name: str) -> str:
    """
    Adjusts the filename for rotated logs before compression.

    Args:
        default_name (str): The default rotated filename

    Returns:
        str: Modified filename (without .gz, handled in `compress_old_log`)
    """
    return default_name
