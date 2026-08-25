import os
import fitz
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_pdf():
    raw_dir = os.path.abspath("data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    doc_path = os.path.join(raw_dir, "citation_sample.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "OmniBrain Citation Test: Revenue for FY2023 increased by 42%.")
    doc.save(doc_path)
    doc.close()
    yield
    if os.path.exists(doc_path):
        os.remove(doc_path)

def test_citation_endpoint_success():
    response = client.get("/api/v1/citations/citation_sample.pdf/1")
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "citation_sample.pdf"
    assert data["page_number"] == 1
    assert "Revenue for FY2023" in data["snippet"]

def test_citation_endpoint_missing_file():
    response = client.get("/api/v1/citations/non_existent_file.pdf/1")
    assert response.status_code == 404

def test_self_rag_query_execution():
    payload = {
        "question": "What was the growth trend in revenue?",
        "top_k": 2
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "retried" in data
    assert isinstance(data["retried"], bool)
    assert "low_confidence" in data
    assert isinstance(data["low_confidence"], bool)
    assert "answer" in data


def test_self_rag_retry_triggered_on_low_confidence_query():
    """
    Validates Self-RAG self-correction:
    Seeds a query guaranteed to yield low confidence (< SIMILARITY_CONFIDENCE_THRESHOLD)
    or empty chunks on the initial pass, asserting that:
    1. The Self-RAG retry path is actively triggered (retried == True).
    2. The rewritten query is generated and distinct from original question.
    3. The original question is preserved in original_question.
    4. The low_confidence flag is computed and returned as True when confidence remains below threshold.
    """
    payload = {
        "question": "What are the cryogenic thermodynamics parameters in quantum helium systems?",
        "document_id": "unindexed_dummy_document.pdf",
        "top_k": 2
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["retried"] is True, "Expected Self-RAG loop to trigger retry on low confidence query."
    assert data["rewritten_query"] is not None, "Expected rewritten_query to be populated."
    assert data["rewritten_query"] != payload["question"], "Rewritten query should differ from original question."
    assert data["original_question"] == payload["question"], "Original question should be preserved."
    assert data["low_confidence"] is True, "Expected low_confidence flag to be True when chunks are missing/low confidence."
    assert data["status"] == "SUCCESS"
    assert "detailed summary and key points" in data["rewritten_query"] or len(data["rewritten_query"]) > len(payload["question"])