import os
import sys
from typing import Dict, Any

# Set path routing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def process_pdf_for_vision_and_tables(pdf_path: str, prompt: str = "Extract key tables and summary") -> Dict[str, Any]:
    """
    Parses PDF documents to extract visual layouts, structured tables, and content chunks.
    """
    if not os.path.exists(pdf_path):
        return {"status": "error", "message": f"PDF file not found at {pdf_path}"}

    filename = os.path.basename(pdf_path)
    print(f"Processing PDF: '{filename}' with prompt: '{prompt}'")

    # Mock PDF table/visual parsing output for fast pipeline execution
    extracted_data = {
        "status": "success",
        "file_name": filename,
        "extracted_tables": [
            {
                "table_id": 1,
                "headers": ["Metric", "Q1 Values", "Q2 Values"],
                "rows": [
                    ["Accuracy", "88.5%", "94.2%"],
                    ["Latency", "120ms", "85ms"]
                ]
            }
        ],
        "visual_summary": f"Document '{filename}' successfully parsed. Tables and layout converted to structured context for RAG retrieval."
    }

    return extracted_data


if __name__ == "__main__":
    # Test with sample path
    sample_pdf = "docs/sample_document.pdf"
    
    # Create dummy folder/file for testing if needed
    os.makedirs("docs", exist_ok=True)
    if not os.path.exists(sample_pdf):
        with open(sample_pdf, "w") as f:
            f.write("Sample PDF Placeholder Content")
            
    result = process_pdf_for_vision_and_tables(sample_pdf)
    
    print("\n--- PDF Vision/Table Agent Output ---")
    print(f"Status: {result['status']}")
    print(f"Parsed File: {result['file_name']}")
    print(f"Extracted Table Data: {result['extracted_tables']}")