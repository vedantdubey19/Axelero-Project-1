"""Tests for app/services/ocr.py"""

from pathlib import Path

import fitz

from app.services.ocr import run_ocr_on_page


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
