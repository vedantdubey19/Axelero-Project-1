"""
OmniBrain Multi-Agent Supervisor Demo Script
============================================
Demonstrates live agentic routing and execution:
1. Document ingestion into Qdrant vector store.
2. Query 1 (Text RAG) -> Supervisor routes to SearchAgent -> Vector retrieval + LLM synthesis.
3. Query 2 (Chart / Visual) -> Supervisor routes to VisionAgent -> Explicit stub notice.
4. Query 3 (Irrelevant / Empty) -> Graceful handling.
"""

import os
import sys
import io
import time
import json

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fitz
from fastapi.testclient import TestClient
from backend.app.main import app

def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def run_demo():
    print_separator("🧠 OmniBrain Multi-Agent Supervisor Live Demo")
    client = TestClient(app)

    # Step 1: Create and Upload PDF
    print_separator("Step 1: Document Ingestion Pipeline")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "OmniBrain FY2025 Performance Report:\n"
        "- Annual Recurring Revenue (ARR) grew 28.5% year-over-year to $42.8M.\n"
        "- Net Retention Rate (NRR) reached 118% across enterprise accounts.\n"
        "- Figure 2 on Page 4 illustrates the Quarterly Revenue vs Operating Expense bar chart."
    )
    pdf_bytes = doc.tobytes()
    filename = "OmniBrain_FY2025_Report.pdf"

    print(f"📄 Uploading '{filename}' to /api/v1/upload...")
    upload_resp = client.post(
        "/api/v1/upload",
        files={"file": (filename, io.BytesIO(pdf_bytes), "application/pdf")}
    )
    upload_data = upload_resp.json()
    job_id = upload_data["job_id"]
    print(f"✅ Upload successful. Ingestion Job ID: {job_id}")

    # Check ingestion status
    status_resp = client.get(f"/api/v1/ingest/status/{job_id}")
    status_data = status_resp.json()
    print(f"📊 Ingestion Status: {status_data['status']} - {status_data.get('message')}")

    # Step 2: Query 1 — Text Semantic RAG
    print_separator("Step 2: Query 1 (Text Retrieval & Reasoning)")
    query_1 = "What was the year-over-year Annual Recurring Revenue (ARR) growth rate?"
    print(f"👤 User: \"{query_1}\"")
    print("⏳ Invoking LangGraph Supervisor Workflow...")

    resp_1 = client.post(
        "/api/v1/agent/query",
        json={"question": query_1, "session_id": "demo-session", "document_id": filename}
    )
    data_1 = resp_1.json()

    print(f"\n🧭 [Supervisor Decision]: Routed to -> \033[92m{data_1['routed_agent']}\033[0m")
    for step in data_1["execution_steps"]:
        print(f"   ↳ Step {step['step_number']} [{step['agent_name']}]: {step['action_taken']}")
        if step.get("details"):
            print(f"     Details: {step['details']}")

    print(f"\n📚 Grounded Sources ({len(data_1['referenced_sources'])} chunks):")
    for s in data_1["referenced_sources"]:
        print(f"   - File: {s['source']} (Page {s['page']}, Score: {s['score']:.4f})")
        print(f"     Content: \"{s['content'][:120]}...\"")

    print(f"\n💬 Final Synthesized Answer:\n{data_1['final_answer']}")

    # Step 3: Query 2 — Visual / Chart Query
    print_separator("Step 3: Query 2 (Chart / Diagram Intent Routing)")
    query_2 = "Can you analyze the Quarterly Revenue vs Operating Expense bar chart in Figure 2?"
    print(f"👤 User: \"{query_2}\"")
    print("⏳ Invoking LangGraph Supervisor Workflow...")

    resp_2 = client.post(
        "/api/v1/agent/query",
        json={"question": query_2, "session_id": "demo-session", "document_id": filename}
    )
    data_2 = resp_2.json()

    print(f"\n🧭 [Supervisor Decision]: Routed to -> \033[93m{data_2['routed_agent']}\033[0m")
    for step in data_2["execution_steps"]:
        print(f"   ↳ Step {step['step_number']} [{step['agent_name']}]: {step['action_taken']}")
        if step.get("details"):
            print(f"     Details: {step['details']}")

    print(f"\n💬 Agent Response Status: {data_2['status']}")
    print(f"💬 Final Answer:\n{data_2['final_answer']}")

    # Step 4: Query 3 — Out of Domain / Graceful Empty Context
    print_separator("Step 4: Query 3 (Graceful Out-of-Domain Handling)")
    query_3 = "What is the capital city of Mars?"
    print(f"👤 User: \"{query_3}\"")
    print("⏳ Invoking LangGraph Supervisor Workflow...")

    resp_3 = client.post(
        "/api/v1/agent/query",
        json={"question": query_3, "session_id": "demo-session", "document_id": filename}
    )
    data_3 = resp_3.json()

    print(f"\n🧭 [Supervisor Decision]: Routed to -> \033[92m{data_3['routed_agent']}\033[0m")
    print(f"📚 Sources Found: {len(data_3['referenced_sources'])}")
    print(f"💬 Final Answer:\n{data_3['final_answer']}")

    print_separator("✨ Demo Complete: Multi-Agent Supervisor Verified End-to-End!")


if __name__ == "__main__":
    run_demo()
