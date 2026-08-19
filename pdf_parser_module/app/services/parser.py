"""
Core parsing orchestrator.

This is the central service that ties every other service together:
for each page it decides whether to use the native text layer or
fall back to OCR, then extracts images and tables, and finally
assembles everything into a single ParseResult.
"""

import time
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import CorruptedPDFError, PasswordProtectedPDFError, PDFParserError
from app.core.logger import logger
from app.models.schemas import PageResult, ParseResult
from app.services import metadata as metadata_service
from app.services import ocr as ocr_service
from app.services import table_extractor as table_service
from app.services.image_extractor import extract_images_from_page
from app.utils.file_utils import ensure_output_subfolder
from app.utils.pdf_utils import get_page_text, open_pdf, page_has_meaningful_text


def parse_pdf(file_id: str, file_path: Path, original_filename: str) -> ParseResult:
    """
    Run the full parsing pipeline on a single PDF file.

    Pipeline, per page:
        1. Try native text extraction (PyMuPDF).
        2. If the text is too short/empty, assume the page is scanned
           and run OCR instead.
        3. Extract embedded images.
        4. Extract tables.

    Args:
        file_id: unique ID assigned to this file at upload time.
        file_path: path to the PDF on disk.
        original_filename: the filename as the user originally uploaded it,
            kept only for display purposes in the final JSON.

    Returns:
        A ParseResult containing metadata and a per-page breakdown of
        text, images, and tables.

    Raises:
        PDFParserError (or a subclass): if the document cannot be
        opened at all (corrupted or password protected).
    """
    start_time = time.perf_counter()
    logger.info(f"Starting parse for file_id={file_id} ({original_filename})")

    try:
        document = open_pdf(file_path)
    except (CorruptedPDFError, PasswordProtectedPDFError):
        # Re-raise as-is; the API layer knows how to translate these
        # into meaningful HTTP responses.
        raise

    doc_metadata = metadata_service.extract_metadata(document, file_path)
    metadata_service.save_metadata_json(file_id, doc_metadata)

    image_output_folder = ensure_output_subfolder(settings.IMAGE_OUTPUT_DIR, file_id)
    table_output_folder = ensure_output_subfolder(settings.TABLE_OUTPUT_DIR, file_id)
    text_output_folder = ensure_output_subfolder(settings.TEXT_OUTPUT_DIR, file_id)

    # Tables are extracted once for the whole document before page assembly.
    tables_by_page = table_service.extract_all_tables(
        file_path, document.page_count, table_output_folder
    )

    pages: list[PageResult] = []

    for page_index in range(document.page_count):
        page_number = page_index + 1
        page = document[page_index]

        page_text = get_page_text(page)
        ocr_used = False

        if not page_has_meaningful_text(page, page_text, settings.OCR_TEXT_THRESHOLD):
            logger.info(f"Page {page_number} has little/no text layer, running OCR")
            # Set before attempting OCR, not just on success: this page
            # was determined to need OCR either way, and a downstream
            # consumer should be able to tell "OCR was the intended
            # path but failed" apart from "this page had a normal
            # native text layer that happened to be empty".
            ocr_used = True
            try:
                page_text = ocr_service.run_ocr_on_page(page, page_number)
            except PDFParserError as error:
                # If OCR itself fails, we do not abort the whole
                # document - we log it and continue with empty text
                # for this page, so the rest of the PDF still parses.
                logger.error(f"Continuing without text for page {page_number}: {error}")
                page_text = ""

        # Save the page's text to its own file for easy inspection,
        # in addition to embedding it in the final JSON.
        text_file = text_output_folder / f"page_{page_number}.txt"
        text_file.write_text(page_text, encoding="utf-8")

        images = extract_images_from_page(document, page, page_number, image_output_folder)
        tables = tables_by_page.get(page_number, [])

        pages.append(
            PageResult(
                page_number=page_number,
                text=page_text,
                ocr_used=ocr_used,
                images=images,
                tables=tables,
            )
        )

    document.close()

    elapsed = round(time.perf_counter() - start_time, 3)
    logger.info(f"Finished parsing file_id={file_id} in {elapsed}s")

    return ParseResult(
        file_id=file_id,
        file_name=original_filename,
        metadata=doc_metadata,
        pages=pages,
        processing_time_seconds=elapsed,
    )
