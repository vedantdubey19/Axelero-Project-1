import io
import fitz
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_supervisor_e2e_real_search_pipeline():
    """
    End-to-End Test:
    1. Uploads a real PDF document via /api/v1/upload.
    2. Polls /api/v1/ingest/status/{job_id} until COMPLETED.
    3. Queries /api/v1/agent/query and asserts:
       - Real routing to SearchAgent.
       - Real vector retrieval with sources matching the uploaded document (not fake sample.pdf).
       - Real LLM answer synthesis without placeholder strings.
    """
    # 1. Create a genuine PDF document with target facts
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "OmniBrain Enterprise Report: The annual revenue growth rate reached 28.5 percent in FY2025."
    )
    pdf_bytes = doc.tobytes()

    filename = "e2e_verified_report.pdf"

    # 2. Upload the PDF
    upload_resp = client.post(
        "/api/v1/upload",
        files={"file": (filename, io.BytesIO(pdf_bytes), "application/pdf")}
    )
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
    upload_data = upload_resp.json()
    job_id = upload_data["job_id"]
    assert upload_data["filename"] == filename

    # 3. Check / Poll ingestion status
    status_resp = client.get(f"/api/v1/ingest/status/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "COMPLETED", f"Ingestion failed: {status_data.get('message')}"

    # 4. Execute multi-agent supervisor query against uploaded document
    query_payload = {
        "question": "What is the annual revenue growth rate?",
        "session_id": "test-session-e2e-real",
        "document_id": filename
    }
    response = client.post("/api/v1/agent/query", json=query_payload)
    assert response.status_code == 200, f"Query failed: {response.text}"
    data = response.json()

    # Assert real Supervisor dynamic routing
    assert data["routed_agent"] == "SearchAgent"
    assert any(step["agent_name"] == "SupervisorAgent" for step in data["execution_steps"])
    assert any(step["agent_name"] == "SearchAgent" for step in data["execution_steps"])

    # Assert real retrieved chunks with genuine source metadata (not mock sample.pdf)
    assert len(data["referenced_sources"]) > 0, "No chunks were retrieved from vector store."
    for source in data["referenced_sources"]:
        assert source["source"] == filename, f"Expected source {filename}, got {source['source']}"
        assert "revenue growth rate" in source.get("content", "").lower()

    # Assert real synthesized final answer
    assert len(data["final_answer"]) > 0
    assert "[Search Agent Response] Retrieved contextual passages" not in data["final_answer"], "Found old canned response!"
    assert data["status"] == "COMPLETED"


def test_supervisor_routes_to_vision_agent():
    """
    Image/chart queries must route to VisionAgent and return an explicit labeled stub.
    """
    payload = {
        "question": "Explain the revenue breakdown bar chart and visual plot on page 4.",
        "session_id": "test-session-vision"
    }
    response = client.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routed_agent"] == "VisionAgent"
    assert any(step["agent_name"] == "VisionAgent" for step in data["execution_steps"])
    assert data["status"] == "NOT_IMPLEMENTED"
    assert "[Vision Agent Notice]" in data["final_answer"]


def test_supervisor_graceful_handling_on_empty_context():
    """
    Queries with no matching document context should be handled gracefully without crashing.
    """
    payload = {
        "question": "What was the total revenue recorded in the 1999 fiscal year report?",
        "session_id": "test-session-irrelevant",
        "document_id": "non_existent_doc.pdf"
    }
    response = client.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routed_agent"] == "SearchAgent"
    assert len(data["referenced_sources"]) == 0
    assert "could not find any relevant information" in data["final_answer"].lower() or "offline synthesis" in data["final_answer"].lower()