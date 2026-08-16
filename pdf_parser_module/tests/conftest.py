"""
Shared pytest fixtures.

`conftest.py` is automatically discovered by pytest, so fixtures
defined here are available to every test file in this folder without
needing to import them manually.
"""

from pathlib import Path

import fitz
import pytest
@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """
    Build a small, real, valid one-page PDF on the fly using PyMuPDF,
    so tests do not depend on a fixture file being present in the
    repository. This keeps the test suite fully self-contained.
    """
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "This is a sample PDF used for testing.")
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture
def blank_pdf_path(tmp_path: Path) -> Path:
    """A PDF page with no text at all, used to test the OCR fallback path."""
    pdf_path = tmp_path / "blank.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture
def empty_pdf_path(tmp_path: Path) -> Path:
    """
    A structurally valid, zero-page PDF file.

    PyMuPDF refuses to save a zero-page document - document.save()
    raises "ValueError: cannot save with zero pages" regardless of how
    the document reached that state, so a zero-page PDF cannot be
    produced by building one in fitz and calling save(). Instead we
    write the minimal valid PDF byte structure directly: a Catalog
    object pointing at a Pages tree with an empty Kids array and
    Count 0. This is exactly what a genuine zero-page PDF looks like
    on disk (this is a valid, well-formed PDF - a Pages tree with no
    Kids is explicitly permitted by the PDF spec), so it still tests
    parse_pdf's zero-page handling against a real file on disk rather
    than skipping or faking the scenario.
    """
    catalog_object = b"<< /Type /Catalog /Pages 2 0 R >>"
    pages_object = b"<< /Type /Pages /Kids [] /Count 0 >>"
    objects = [catalog_object, pages_object]

    body = bytearray()
    body += b"%PDF-1.4\n"

    # Track the exact byte offset of each object as we write it, so
    # the xref table below points to genuinely correct positions
    # rather than guessed/hardcoded ones.
    offsets = []
    for index, obj_content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{index} 0 obj\n".encode("ascii")
        body += obj_content + b"\nendobj\n"

    xref_offset = len(body)
    object_count = len(objects) + 1  # +1 for the mandatory free-list head

    body += f"xref\n0 {object_count}\n".encode("ascii")
    body += b"0000000000 65535 f \n"
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n".encode("ascii")

    body += b"trailer\n"
    body += f"<< /Size {object_count} /Root 1 0 R >>\n".encode("ascii")
    body += b"startxref\n"
    body += f"{xref_offset}\n".encode("ascii")
    body += b"%%EOF"

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(bytes(body))
    return pdf_path


@pytest.fixture
def password_protected_pdf_path(tmp_path: Path) -> Path:
    """A PDF that requires a password to open."""
    pdf_path = tmp_path / "protected.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Secret content")
    document.save(pdf_path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret123")
    document.close()
    return pdf_path


@pytest.fixture
def corrupted_pdf_path(tmp_path: Path) -> Path:
    """A file with a .pdf extension whose content is not a valid PDF at all."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nthis is not actually valid pdf structure")
    return pdf_path


@pytest.fixture
def scanned_pdf_path(tmp_path: Path) -> Path:
    """
    A PDF page containing only an image of text, with no real text
    layer at all - this is what a scanned document looks like to
    PyMuPDF, and should force the OCR fallback path in parse_pdf.
    """
    from PIL import Image, ImageDraw

    # Build a plain image containing rendered text, standing in for a
    # scanned page (no font/encoding tricks, just pixels).
    image = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "Scanned page sample text", fill="black")
    image_path = tmp_path / "scanned_source.png"
    image.save(image_path)

    pdf_path = tmp_path / "scanned.pdf"
    document = fitz.open()
    page = document.new_page()
    # Insert the image covering the page and add no text at all, so
    # PyMuPDF's text extraction finds nothing and OCR must run.
    page.insert_image(page.rect, filename=str(image_path))
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture
def multi_page_pdf_path(tmp_path: Path) -> Path:
    """A PDF with many pages, standing in for a larger real-world document."""
    pdf_path = tmp_path / "multi_page.pdf"
    document = fitz.open()
    for page_index in range(20):
        page = document.new_page()
        page.insert_text((72, 72), f"This is page number {page_index + 1} of the document.")
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture
def image_heavy_pdf_path(tmp_path: Path) -> Path:
    """A PDF with several embedded images across two pages."""
    from PIL import Image

    document = fitz.open()
    for page_index in range(2):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {page_index + 1} with embedded images")
        for image_index in range(2):
            image = Image.new(
                "RGB", (100, 100), color=(image_index * 40, 100, 150)
            )
            image_path = tmp_path / f"img_{page_index}_{image_index}.png"
            image.save(image_path)
            # Place each image in a different spot so they don't overlap.
            rect = fitz.Rect(50 + image_index * 120, 150, 150 + image_index * 120, 250)
            page.insert_image(rect, filename=str(image_path))
    pdf_path = tmp_path / "image_heavy.pdf"
    document.save(pdf_path)
    document.close()
    return pdf_path
