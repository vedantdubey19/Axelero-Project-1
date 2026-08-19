# OmniBrain PDF Document Processing Module

Document-processing component for OmniBrain. This module handles the work assigned to Humera through August 14:

- PDF text extraction with PyMuPDF
- Table extraction with pdfplumber
- Embedded image extraction with PyMuPDF
- OCR fallback for scanned/image-only pages using Tesseract
- Automated tests for text, table, image, OCR, and parser behavior

## Setup

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR separately and ensure it is available on PATH. On Windows, set `TESSERACT_CMD` in `.env` if needed.

## Tests

Run:

```bash
pytest -v
```

## Sample fixtures

`sample_pdfs/` contains small fixtures for text/image/table/OCR scenarios. Runtime uploads and generated extraction output are intentionally excluded from Git.
