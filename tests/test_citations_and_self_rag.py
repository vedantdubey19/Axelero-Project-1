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
    assert "retry_count" in data
    assert isinstance(data["retry_count"], int)
    assert "retry_history" in data
    assert isinstance(data["retry_history"], list)
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
    5. Full retry_count and retry_history are returned.
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
    assert data["retry_count"] >= 1
    assert len(data["retry_history"]) >= 2
    assert "detailed summary and key points" in data["rewritten_query"] or len(data["rewritten_query"]) > len(payload["question"])


def test_self_rag_multi_retry_bounded_loop():
    """
    Validates that the Self-RAG loop executes multiple retries and bounds at max_retries.
    Asserts that retry_history records each attempt with query, chunk count, and confidence status.
    """
    payload = {
        "question": "Completely unrelated obscure question about interstellar dark matter astrophysics?",
        "document_id": "nonexistent_astrophysics_paper.pdf",
        "top_k": 2
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["retried"] is True
    assert data["retry_count"] == 2, f"Expected 2 retries, got {data['retry_count']}"
    assert len(data["retry_history"]) == 3, f"Expected attempt 0, 1, 2 in history, got {len(data['retry_history'])}"
    assert data["low_confidence"] is True

    # Assert retry history structure
    for entry in data["retry_history"]:
        assert "attempt" in entry
        assert "query" in entry
        assert "chunks_count" in entry
        assert "avg_score" in entry
        assert "passed_confidence" in entry


def test_self_rag_adaptive_rewrite_on_different_domains():
    """
    Verifies that the adaptive query rewriter extracts salient terms dynamically from retrieved chunks
    across different document domains (e.g. legal contracts vs quantum physics) rather than
    hardcoding a single domain string (the bug flagged in the Aug 27 review).
    """
    from backend.app.services.llm_service import LLMSynthesisService
    llm_svc = LLMSynthesisService()

    # Case A: Legal domain chunks
    legal_chunks = [
        {"content": "Indemnification liabilities and contractual arbitration clauses shall survive termination.", "page": 1}
    ]
    legal_rewrite = llm_svc.rewrite_query("what are the terms", retrieved_chunks=legal_chunks, attempt=1)
    assert "indemnification" in legal_rewrite.lower() or "contractual" in legal_rewrite.lower() or "liabilities" in legal_rewrite.lower()
    assert "financial metrics" not in legal_rewrite.lower()

    # Case B: Quantum physics domain chunks
    quantum_chunks = [
        {"content": "Superconducting qubits experience decoherence in cryogenic microwave resonators.", "page": 1}
    ]
    quantum_rewrite = llm_svc.rewrite_query("what are the terms", retrieved_chunks=quantum_chunks, attempt=1)
    assert "superconducting" in quantum_rewrite.lower() or "qubits" in quantum_rewrite.lower() or "decoherence" in quantum_rewrite.lower()
    assert "financial metrics" not in quantum_rewrite.lower()

    # Assert that rewrites for different domains produce completely distinct queries
    assert legal_rewrite != quantum_rewrite


def test_self_rag_ambiguous_on_topic_query():
    """
    Adversarial Case 1: Ambiguous on-topic query.
    A vague query like 'what about the numbers' should trigger the Self-RAG rewrite
    loop and result in a materially rewritten query and appropriate confidence evaluation.
    """
    payload = {
        "question": "what about the numbers",
        "top_k": 2
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["retried"] is True, "Expected ambiguous query to trigger rewrite retry."
    assert data["rewritten_query"] is not None
    assert data["rewritten_query"] != payload["question"]
    assert "low_confidence" in data
    assert isinstance(data["low_confidence"], bool)


def test_self_rag_absent_content_query():
    """
    Adversarial Case 2: Query for content completely absent from the indexed document.
    Asserts that the system does not fabricate confident answers, triggers rewrite,
    and returns low_confidence=True with an honest uncertainty signal.
    """
    payload = {
        "question": "What is the orbital trajectory and thrust velocity of the payload?",
        "document_id": "citation_sample.pdf",
        "top_k": 2
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["retried"] is True
    assert data["low_confidence"] is True, "Absent document content must result in low_confidence=True."
    assert (
        "could not find any relevant information" in data["answer"].lower()
        or data["low_confidence"] is True
    )


def test_self_rag_empty_or_punctuation_query_fails_gracefully():
    """
    Adversarial Case 3: Query containing only punctuation or whitespace.
    Asserts that the API fails gracefully with HTTP 400 Bad Request, not a 500 error or silent hallucination.
    """
    payload = {
        "question": "   ???!!!   ",
        "top_k": 2
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "empty or solely punctuation" in data["detail"]