import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_supervisor_routes_to_search_agent():
    """Text/conceptual queries must route to SearchAgent."""
    payload = {
        "question": "What is the annual revenue growth rate?",
        "session_id": "test-session-search"
    }
    response = client.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routed_agent"] == "SearchAgent"
    assert any(step["agent_name"] == "SearchAgent" for step in data["execution_steps"])

def test_supervisor_routes_to_vision_agent():
    """Image/chart queries must route to VisionAgent."""
    payload = {
        "question": "Explain the bar chart and visual plot on page 4.",
        "session_id": "test-session-vision"
    }
    response = client.post("/api/v1/agent/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routed_agent"] == "VisionAgent"
    assert any(step["agent_name"] == "VisionAgent" for step in data["execution_steps"])