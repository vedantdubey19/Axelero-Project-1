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
    assert "answer" in data