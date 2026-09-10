import pytest
import concurrent.futures
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.retriever_service import RetrieverService, preload_embedder

def test_concurrent_embedder_direct_calls():
    """Verify that multiple threads calling retrieve_relevant_chunks concurrently do not crash."""
    preload_embedder()
    service = RetrieverService()
    queries = [
        f"Query {i}: What are the financial projections for Q{i % 4 + 1}?"
        for i in range(15)
    ]

    errors = []
    def worker(q):
        try:
            res = service.retrieve_relevant_chunks(q, top_k=2)
            return res
        except Exception as e:
            errors.append(str(e))
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, queries))

    assert len(errors) == 0, f"Errors encountered during concurrent retrieval: {errors}"
    assert len(results) == 15


def test_concurrent_api_queries():
    """Verify 10+ simultaneous requests to /api/v1/query execute without meta-tensor collisions."""
    preload_embedder()
    client = TestClient(app)
    queries = [
        f"Concurrent query {i} testing revenue growth and metrics"
        for i in range(12)
    ]

    responses = []
    errors = []

    def api_worker(q):
        try:
            resp = client.post("/api/v1/query", json={"question": q, "top_k": 2})
            return resp
        except Exception as e:
            errors.append(str(e))
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        responses = list(executor.map(api_worker, queries))

    assert len(errors) == 0, f"Exceptions raised in threads: {errors}"
    status_codes = [r.status_code for r in responses if r is not None]
    assert all(code == 200 for code in status_codes), f"Non-200 responses: {status_codes}"
    for r in responses:
        body = r.json()
        assert "answer" in body
        assert "query_id" in body
