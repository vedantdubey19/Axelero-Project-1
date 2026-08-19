"""
Application configuration.

All paths, limits, and environment-driven settings live here so that
no other module hardcodes a file path or a magic number. Values are
read from environment variables when available, with sensible
defaults for local development.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if one exists in the project root.
# This does nothing if .env is missing, so it is safe to call always.
load_dotenv()

# Root of the project, resolved dynamically so the app works no matter
# where it is cloned to on disk.
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent


class Settings:
    """
    Groups every configurable value used across the application.

    Using a class (instead of loose module-level variables) makes it
    possible to import a single `settings` object anywhere in the
    codebase and access everything through it, e.g. settings.UPLOAD_DIR.
    """

    # --- Folder locations -------------------------------------------------
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
    OUTPUT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "output")

    TEXT_OUTPUT_DIR: Path = OUTPUT_DIR / "text"
    IMAGE_OUTPUT_DIR: Path = OUTPUT_DIR / "images"
    TABLE_OUTPUT_DIR: Path = OUTPUT_DIR / "tables"
    METADATA_OUTPUT_DIR: Path = OUTPUT_DIR / "metadata"
    JSON_OUTPUT_DIR: Path = OUTPUT_DIR / "json"

    # --- Upload validation --------------------------------------------------
    ALLOWED_EXTENSIONS: set[str] = {".pdf"}
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

    # --- OCR settings ---------------------------------------------------
    # Minimum number of characters a page must contain before we trust
    # the extracted text instead of falling back to OCR.
    OCR_TEXT_THRESHOLD: int = int(os.getenv("OCR_TEXT_THRESHOLD", "20"))

    # DPI used when rasterizing a PDF page into an image for OCR.
    # Higher DPI improves OCR accuracy but is slower.
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))

    # Path to the tesseract executable. On Windows this typically needs
    # to be set explicitly, e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
    TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD")

    # --- Logging -------------------------------------------------------
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "parser.log"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create every directory this app depends on if it does not
        already exist. Called once at application startup so the rest
        of the code can safely assume these folders are present.
        """
        directories = [
            cls.UPLOAD_DIR,
            cls.TEXT_OUTPUT_DIR,
            cls.IMAGE_OUTPUT_DIR,
            cls.TABLE_OUTPUT_DIR,
            cls.METADATA_OUTPUT_DIR,
            cls.JSON_OUTPUT_DIR,
            cls.LOG_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Single shared instance imported by the rest of the application.
settings = Settings()
