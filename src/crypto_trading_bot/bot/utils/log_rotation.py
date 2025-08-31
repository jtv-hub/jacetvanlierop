"""
log_rotation.py

Provides rotating log handlers and optional compression for the crypto trading bot.
"""

import os
import gzip
import shutil
from logging.handlers import RotatingFileHandler
import logging

LOG_DIR = "logs"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5


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
