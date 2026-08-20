"""
OCR service.

When a page has no usable text layer (typically because it is a
scanned image rather than digitally-created text), we rasterize the
page into an image and run Tesseract OCR on it to recover the text.
"""

import fitz
import pytesseract

try:
    from pdf_parser_module.app.core.config import settings
    from pdf_parser_module.app.core.exceptions import OCRProcessingError
    from pdf_parser_module.app.core.logger import logger
    from pdf_parser_module.app.utils.image_utils import bytes_to_pil_image
    from pdf_parser_module.app.utils.pdf_utils import render_page_to_image
except ImportError:
    from app.core.config import settings
    from app.core.exceptions import OCRProcessingError
    from app.core.logger import logger
    from app.utils.image_utils import bytes_to_pil_image
    from app.utils.pdf_utils import render_page_to_image

# If a custom tesseract path was provided (common on Windows, where it
# is not automatically added to PATH), configure pytesseract to use it.
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def run_ocr_on_page(page: fitz.Page, page_number: int) -> str:
    """
    Perform OCR on a single PDF page and return the recognized text.

    Args:
        page: the PyMuPDF page object to process.
        page_number: 1-indexed page number, used only for logging.

    Returns:
        The text recognized by Tesseract, stripped of leading/trailing
        whitespace. Returns an empty string if OCR finds nothing.

    Raises:
        OCRProcessingError: if Tesseract fails to run at all (for
        example, if it is not installed or not found on PATH).
    """
    try:
        image_bytes = render_page_to_image(page, dpi=settings.OCR_DPI)
        image = bytes_to_pil_image(image_bytes)

        recognized_text = pytesseract.image_to_string(image)
        recognized_text = recognized_text.strip()

        logger.info(
            f"OCR completed for page {page_number}: "
            f"{len(recognized_text)} characters recognized"
        )
        return recognized_text

    except pytesseract.TesseractNotFoundError as error:
        logger.error(
            "Tesseract executable not found. Install it and/or set "
            "TESSERACT_CMD in your .env file."
        )
        raise OCRProcessingError(
            "Tesseract OCR engine is not installed or not found on PATH."
        ) from error

    except Exception as error:
        logger.error(f"OCR failed on page {page_number}: {error}")
        raise OCRProcessingError(f"OCR failed on page {page_number}") from error
