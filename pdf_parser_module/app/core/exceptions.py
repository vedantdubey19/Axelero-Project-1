"""
Custom exception types for the PDF Parser Module.

Using specific exception classes (instead of raising generic
Exception or ValueError everywhere) lets the API layer catch each
failure case and return an appropriate HTTP status code and message,
instead of a generic 500 error for every possible problem.
"""


class PDFParserError(Exception):
    """Base class for every custom exception raised by this module."""


class InvalidFileTypeError(PDFParserError):
    """Raised when an uploaded file is not a PDF."""


class FileTooLargeError(PDFParserError):
    """Raised when an uploaded file exceeds the configured size limit."""


class FileNotFoundInStorageError(PDFParserError):
    """Raised when a requested file ID does not exist in uploads/."""


class InvalidFileIdError(PDFParserError):
    """
    Raised when a file_id path parameter does not match the expected
    format. file_id values are always generated internally by this
    service (see generate_file_id), so anything that does not match
    that format is either a client mistake or a path traversal
    attempt, and must be rejected before it is used to build a
    filesystem path.
    """


class CorruptedPDFError(PDFParserError):
    """Raised when a PDF file cannot be opened or parsed at all."""


class PasswordProtectedPDFError(PDFParserError):
    """Raised when a PDF requires a password to open."""


class OCRProcessingError(PDFParserError):
    """Raised when OCR fails on a page that required it."""


class TableExtractionError(PDFParserError):
    """Raised when table extraction fails for a document or page."""


class MetadataExtractionError(PDFParserError):
    """Raised when metadata cannot be read from a PDF."""
