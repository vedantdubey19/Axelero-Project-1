"""
Low-level PDF helper functions built on top of PyMuPDF (fitz).

These functions wrap raw PyMuPDF calls with error handling so the
higher-level services (parser.py, ocr.py, etc.) never have to deal
with fitz exceptions directly.
"""

from pathlib import Path

import fitz  # PyMuPDF

try:
    from pdf_parser_module.app.core.exceptions import CorruptedPDFError, PasswordProtectedPDFError
    from pdf_parser_module.app.core.logger import logger
except ImportError:
    from app.core.exceptions import CorruptedPDFError, PasswordProtectedPDFError
    from app.core.logger import logger


def open_pdf(path: Path) -> fitz.Document:
    """
    Open a PDF file and return a PyMuPDF Document object.

    Raises:
        PasswordProtectedPDFError: if the PDF requires a password.
        CorruptedPDFError: if the file cannot be parsed as a PDF at all.
    """
    try:
        document = fitz.open(path)
    except Exception as error:
        logger.error(f"Failed to open PDF at {path}: {error}")
        raise CorruptedPDFError(f"Could not open PDF file: {path.name}") from error

    if document.needs_pass:
        logger.warning(f"PDF at {path} is password protected")
        raise PasswordProtectedPDFError(
            f"'{path.name}' is password protected and cannot be parsed."
        )

    return document


def get_page_text(page: fitz.Page) -> str:
    """
    Extract raw text from a single page using PyMuPDF's built-in
    text extraction. This works well for PDFs that contain a real
    text layer (as opposed to scanned image-only pages).
    """
    return page.get_text().strip()


def page_has_meaningful_text(page: fitz.Page, text: str, threshold: int) -> bool:
    """
    Decide whether a page's extracted text is substantial enough to
    trust, or whether the page is likely a scanned image that needs OCR.

    A raw character-length check alone can be fooled: a page that is
    actually a scanned image sometimes still has a handful of stray
    embedded characters (page numbers, watermark text, OCR leftovers
    from a previous processing pass) that add up to more than a few
    characters without being real body text. We combine three cheap
    signals instead of trusting length alone:

        1. Character length - the basic "is there almost nothing here" check.
        2. Word count - a single long garbled token is not real prose.
        3. Text block count - PyMuPDF groups text into layout blocks;
           genuine digital text pages normally have several blocks
           (paragraphs, headers), while noise tends to sit in one.

    This stays intentionally simple - it is a fast heuristic, not a
    document classifier.
    """
    if len(text) < threshold:
        return False

    word_count = len(text.split())
    if word_count < 3:
        return False

    text_blocks = page.get_text("blocks")
    if len(text_blocks) < 1:
        return False

    return True


def render_page_to_image(page: fitz.Page, dpi: int) -> "bytes":
    """
    Rasterize a PDF page into a PNG image, returned as raw bytes.

    This is used in two places: feeding a page image to the OCR
    engine, and (indirectly) for any downstream image-based
    processing. DPI controls resolution: higher DPI means a sharper
    image and better OCR accuracy, at the cost of speed and memory.
    """
    zoom = dpi / 72  # PDF default resolution is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    return pixmap.tobytes("png")
