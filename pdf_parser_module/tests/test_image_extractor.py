"""Tests for app/services/image_extractor.py"""

from pathlib import Path

import fitz

from app.services.image_extractor import extract_images_from_page


def test_extract_images_from_page_with_no_images_returns_empty_list(
    sample_pdf_path: Path, tmp_path: Path
) -> None:
    document = fitz.open(sample_pdf_path)
    page = document[0]
    output_folder = tmp_path / "images"

    images = extract_images_from_page(document, page, page_number=1, output_folder=output_folder)

    assert images == []
    document.close()
