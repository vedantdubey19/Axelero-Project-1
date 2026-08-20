"""
Image helper functions shared by the OCR and image extraction services.
"""

import io
from pathlib import Path

from PIL import Image

try:
    from pdf_parser_module.app.core.logger import logger
except ImportError:
    from app.core.logger import logger


def save_image_bytes(image_bytes: bytes, destination: Path) -> tuple[int, int]:
    """
    Save raw image bytes to disk and return the image's (width, height).

    Using Pillow to open the bytes before saving also acts as a
    validation step: if the bytes are not a real image, this will
    raise, and the caller can decide how to handle that failure.
    """
    image = Image.open(io.BytesIO(image_bytes))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    logger.debug(f"Saved image to {destination} ({image.width}x{image.height})")
    return image.width, image.height


def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes into a Pillow Image object for further processing."""
    return Image.open(io.BytesIO(image_bytes))
