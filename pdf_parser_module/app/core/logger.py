"""
Centralized logging configuration.

Every other module imports `logger` from this file instead of calling
`logging.getLogger(__name__)` directly. This keeps log formatting and
log file location consistent across the whole project, and means the
logging setup only has to be changed in one place.
"""

import sys

from loguru import logger

from app.core.config import settings

# Make sure the logs/ directory actually exists before loguru tries
# to write to it.
settings.ensure_directories()

# Remove the default loguru handler so we can fully control the format.
logger.remove()

# Console handler: readable, colorized output while developing locally.
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)

# File handler: plain text, rotated when it grows too large, and kept
# for a limited number of days so logs do not grow forever.
logger.add(
    settings.LOG_FILE,
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="14 days",
    encoding="utf-8",
)

__all__ = ["logger"]
