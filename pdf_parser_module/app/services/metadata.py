"""
Metadata extraction service.

Reads the document-level info dictionary embedded in a PDF (title,
author, subject, dates, etc.) and normalizes it into our PDFMetadata
schema.
"""

from pathlib import Path

import fitz

try:
    from pdf_parser_module.app.core.config import settings
    from pdf_parser_module.app.core.exceptions import MetadataExtractionError
    from pdf_parser_module.app.core.logger import logger
    from pdf_parser_module.app.models.schemas import PDFMetadata
except ImportError:
    from app.core.config import settings
    from app.core.exceptions import MetadataExtractionError
    from app.core.logger import logger
    from app.models.schemas import PDFMetadata


def extract_metadata(document: fitz.Document, file_path: Path) -> PDFMetadata:
    """
    Build a PDFMetadata object from an open PyMuPDF document.

    Args:
        document: an already-opened PyMuPDF Document.
        file_path: path to the file on disk, used to read its size.

    Returns:
        A populated PDFMetadata instance.
    """
    try:
        info = document.metadata or {}
        file_size = file_path.stat().st_size

        metadata = PDFMetadata(
            title=info.get("title") or None,
            author=info.get("author") or None,
            subject=info.get("subject") or None,
            keywords=info.get("keywords") or None,
            creation_date=info.get("creationDate") or None,
            modification_date=info.get("modDate") or None,
            total_pages=document.page_count,
            file_size_bytes=file_size,
        )

        logger.info(f"Extracted metadata for {file_path.name}: {metadata.total_pages} pages")
        return metadata

    except Exception as error:
        logger.error(f"Metadata extraction failed for {file_path.name}: {error}")
        raise MetadataExtractionError(
            f"Could not extract metadata from '{file_path.name}'"
        ) from error


def save_metadata_json(file_id: str, metadata: PDFMetadata) -> Path:
    """
    Save extracted metadata to its own standalone JSON file under
    output/metadata/<file_id>.json.

    The same metadata is also embedded inside the full parse result
    saved by json_builder, but keeping a standalone copy here lets a
    caller fetch just the metadata off disk directly, without loading
    the (potentially much larger) full parse result first.
    """
    settings.METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = settings.METADATA_OUTPUT_DIR / f"{file_id}.json"
    destination.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    logger.debug(f"Saved standalone metadata to {destination}")
    return destination
