import os
import fitz  # PyMuPDF
from typing import Dict, Any, Optional


class CitationService:
    """
    Extracts exact page text snippets and metadata for frontend citation popups.
    """
    def __init__(self, raw_docs_dir: str = "data/raw"):
        self.raw_docs_dir = os.path.abspath(raw_docs_dir)
        os.makedirs(self.raw_docs_dir, exist_ok=True)

    def get_page_snippet(self, filename: str, page_number: int) -> Optional[Dict[str, Any]]:
        """
        Extracts raw text and bounding info for a specific PDF page.
        """
        file_path = os.path.join(self.raw_docs_dir, os.path.basename(filename))
        if not os.path.exists(file_path):
            return None

        try:
            doc = fitz.open(file_path)
            zero_indexed_page = max(0, page_number - 1)
            if zero_indexed_page >= len(doc):
                doc.close()
                return None

            page = doc[zero_indexed_page]
            text = page.get_text("text").strip()
            total_pages = len(doc)
            doc.close()

            return {
                "filename": os.path.basename(filename),
                "page_number": page_number,
                "total_pages": total_pages,
                "snippet": text if text else "Page contains no extractable raw text.",
                "char_count": len(text)
            }
        except Exception:
            return None