"""Tests for app/services/parser.py"""

from pathlib import Path

import pytest

try:
    from pdf_parser_module.app.core.exceptions import CorruptedPDFError, PasswordProtectedPDFError
    from pdf_parser_module.app.services.parser import parse_pdf
except ImportError:
    from app.core.exceptions import CorruptedPDFError, PasswordProtectedPDFError
    from app.services.parser import parse_pdf


def test_parse_pdf_returns_correct_page_count(sample_pdf_path: Path) -> None:
    result = parse_pdf("test_id_1", sample_pdf_path, "sample.pdf")

    assert result.file_id == "test_id_1"
    assert result.metadata.total_pages == 1
    assert len(result.pages) == 1


def test_parse_pdf_extracts_text_without_ocr(sample_pdf_path: Path) -> None:
    result = parse_pdf("test_id_2", sample_pdf_path, "sample.pdf")

    page = result.pages[0]
    assert "sample PDF" in page.text
    assert page.ocr_used is False


def test_parse_pdf_records_processing_time(sample_pdf_path: Path) -> None:
    result = parse_pdf("test_id_3", sample_pdf_path, "sample.pdf")

    assert result.processing_time_seconds >= 0


def test_parse_pdf_handles_empty_document(empty_pdf_path: Path) -> None:
    """A zero-page PDF should parse successfully with an empty page list,
    not crash."""
    result = parse_pdf("test_id_4", empty_pdf_path, "empty.pdf")

    assert result.metadata.total_pages == 0
    assert result.pages == []


def test_parse_pdf_raises_on_password_protected_file(
    password_protected_pdf_path: Path,
) -> None:
    with pytest.raises(PasswordProtectedPDFError):
        parse_pdf("test_id_5", password_protected_pdf_path, "protected.pdf")


def test_parse_pdf_raises_on_corrupted_file(corrupted_pdf_path: Path) -> None:
    with pytest.raises(CorruptedPDFError):
        parse_pdf("test_id_6", corrupted_pdf_path, "corrupted.pdf")


def test_parse_pdf_falls_back_to_ocr_for_scanned_page(scanned_pdf_path: Path) -> None:
    """
    A page with no real text layer (only an image of text) should be
    detected as scanned and routed through OCR, with ocr_used=True.
    """
    result = parse_pdf("test_id_7", scanned_pdf_path, "scanned.pdf")

    page = result.pages[0]
    assert page.ocr_used is True


def test_parse_pdf_extracts_embedded_images(image_heavy_pdf_path: Path) -> None:
    result = parse_pdf("test_id_8", image_heavy_pdf_path, "image_heavy.pdf")

    total_images = sum(len(page.images) for page in result.pages)
    assert total_images == 4
    assert len(result.pages) == 2


def test_parse_pdf_marks_ocr_used_even_when_ocr_itself_fails(
    scanned_pdf_path: Path, monkeypatch
) -> None:
    """
    If a page is determined to need OCR but the OCR engine itself
    fails (e.g. Tesseract missing/misconfigured on the host), the page
    should still be marked ocr_used=True, with empty text rather than
    crashing the whole document. ocr_used reflects that OCR was the
    determined path for this page, not whether it happened to succeed.
    """
    from app.core.exceptions import OCRProcessingError
    from app.services import parser as parser_module

    def fake_failing_ocr(page, page_number):
        raise OCRProcessingError("Tesseract is not installed or not found on PATH.")

    monkeypatch.setattr(parser_module.ocr_service, "run_ocr_on_page", fake_failing_ocr)

    result = parse_pdf("test_id_10", scanned_pdf_path, "scanned.pdf")

    page = result.pages[0]
    assert page.ocr_used is True
    assert page.text == ""


def test_parse_pdf_handles_multi_page_document(multi_page_pdf_path: Path) -> None:
    """
    Confirms every page in a larger document is processed and that
    page numbering stays correct and sequential from start to finish.
    """
    result = parse_pdf("test_id_9", multi_page_pdf_path, "multi_page.pdf")

    assert len(result.pages) == 20
    assert [p.page_number for p in result.pages] == list(range(1, 21))
    assert "page number 1" in result.pages[0].text
    assert "page number 20" in result.pages[19].text
