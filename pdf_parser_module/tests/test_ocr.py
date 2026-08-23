import shutil
from pathlib import Path

import fitz
import pytest

try:
    from pdf_parser_module.app.services.ocr import run_ocr_on_page
except ImportError:
    from app.services.ocr import run_ocr_on_page

HAVE_TESSERACT = shutil.which("tesseract") is not None


@pytest.mark.skipif(not HAVE_TESSERACT, reason="Tesseract OCR binary not installed on host")
def test_run_ocr_on_blank_page_returns_string(blank_pdf_path: Path) -> None:
    """
    A blank page should not crash OCR - it should simply return an
    empty (or near-empty) string, since there is no text to recognize.
    """
    document = fitz.open(blank_pdf_path)
    page = document[0]

    text = run_ocr_on_page(page, page_number=1)

    assert isinstance(text, str)
    document.close()

