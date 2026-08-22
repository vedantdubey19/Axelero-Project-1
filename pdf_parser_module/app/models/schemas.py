"""
Pydantic models describing the shape of data moving in and out of the API.

FastAPI uses these models to validate request bodies, serialize
responses, and auto-generate the OpenAPI docs at /docs. Keeping them
in one file makes it easy to see the entire public data contract of
this module at a glance.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned right after a PDF has been uploaded."""

    file_id: str = Field(..., description="Unique identifier assigned to the uploaded file")
    original_filename: str = Field(..., description="The filename as provided by the client")
    saved_path: str = Field(..., description="Path where the file was stored on the server")
    size_bytes: int = Field(..., description="Size of the uploaded file in bytes")


class TableData(BaseModel):
    """A single extracted table, represented as rows of string cells."""

    table_index: int = Field(..., description="Position of this table on the page, starting at 0")
    csv_path: str = Field(..., description="Path to the CSV file this table was saved as")
    rows: int = Field(..., description="Number of rows in the extracted table")
    columns: int = Field(..., description="Number of columns in the extracted table")


class ImageData(BaseModel):
    """A single extracted embedded image."""

    image_index: int = Field(..., description="Position of this image on the page, starting at 0")
    image_path: str = Field(..., description="Path to the saved image file")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class PageResult(BaseModel):
    """Everything extracted from a single page of the PDF."""

    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(default="", description="Extracted or OCR-recognized text")
    ocr_used: bool = Field(default=False, description="Whether OCR was required for this page")
    images: list[ImageData] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)


class PDFMetadata(BaseModel):
    """Document-level metadata pulled from the PDF's info dictionary."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    total_pages: int = 0
    file_size_bytes: int = 0


class ParseResult(BaseModel):
    """
    The complete structured output for a parsed PDF.

    This is the object that gets saved to output/json/ and returned
    by the /parse and /json/{id} endpoints. It is also exactly what
    the downstream RAG team will consume, so its shape should not
    change without coordinating with them.
    """

    file_id: str
    file_name: str
    metadata: PDFMetadata
    pages: list[PageResult]
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_seconds: float = 0.0


class HealthResponse(BaseModel):
    """Simple response for the /health endpoint."""

    status: str = "ok"
    service: str = "OmniBrain PDF Parser Module"
