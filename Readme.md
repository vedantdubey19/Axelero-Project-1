# OmniBrain

**Agentic Multi-Modal RAG Orchestrator for Complex Document Reasoning**

[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1c3d5a)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-dc244c)](https://qdrant.tech/)
[![Status](https://img.shields.io/badge/status-in--development-yellow)]()

---

## Overview

Standard Retrieval-Augmented Generation (RAG) pipelines break down when documents mix unstructured text with **financial tables, embedded charts, images, and structured historical data**, or when a query requires **multi-step reasoning across heterogeneous data silos**.

**OmniBrain** is an agentic, multi-modal RAG orchestrator built around a **LangGraph supervisor architecture**. It routes incoming queries to the right specialist agent — semantic search, SQL, or vision — synthesizes their outputs, and returns a fully cited, grounded response instead of a single-shot LLM guess.

### Use Case

A quantitative analyst uploads a **500-page corporate financial PDF**. OmniBrain:
1. Parses embedded tables and charts using a Vision-Language Model (VLM)
2. Retrieves relevant semantic text chunks from a vector database
3. Queries historical stock data via a Text-to-SQL agent
4. Synthesizes all three outputs into a single, cited **investment memo** — with every claim traceable back to its source page, table, or chart

---

## System Architecture

```
                         ┌───────────────────────────┐
                         │   User Query (Streamlit)   │
                         └─────────────┬───────────────┘
                                       │
                              ┌────────▼─────────┐
                              │  LangGraph        │
                              │  Supervisor Node   │
                              └───┬───────┬────────┘
                  ┌───────────────┘        └───────────────┐
                  │                                        │
         ┌────────▼─────────┐  ┌──────────────┐   ┌────────▼─────────┐
         │   Search Agent    │  │  SQL Agent    │   │  Vision Agent     │
         │  (Qdrant / FAISS) │  │ (Text-to-SQL) │   │ (GPT-4o / LLaVA)  │
         └────────┬─────────┘  └───────┬───────┘   └────────┬─────────┘
                  │                    │                     │
                  └───────────┬────────┴──────────┬──────────┘
                              │                    │
                     ┌────────▼────────────────────▼────────┐
                     │      Self-RAG Correction Loop          │
                     │  (re-query on irrelevant retrieval)    │
                     └────────────────┬────────────────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │  Guardrails (NeMo)         │
                          │  + Evaluation (Langfuse)   │
                          └────────────┬─────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  Cited Synthesized    │
                            │  Response (FastAPI)   │
                            └───────────────────────┘
```

---

## Key Modules

| Module | Stack | Responsibility |
|---|---|---|
| **Agentic Orchestrator** | LangGraph | Manages state, memory, and routing across Search, SQL, and Vision agents |
| **Multi-Modal Retrieval** | Qdrant / FAISS + CLIP | Stores and retrieves text chunks and image embeddings via semantic similarity |
| **Vision-Language Model** | GPT-4o / LLaVA | Extracts and reasons over charts, graphs, and visual tables |
| **Text-to-SQL Agent** | LangChain SQL toolkit | Converts natural-language queries into SQL against historical structured data |
| **Evaluation & Guardrails** | Langfuse + NeMo Guardrails | Monitors for toxicity/hallucination, enforces grounding in retrieved context, tracks latency & token usage |

---

## Tech Stack

- **Orchestration:** LangGraph, LangChain
- **Backend:** FastAPI (async document upload & query endpoints)
- **Frontend:** Streamlit (agent thought-process viewer, image + citation rendering)
- **Vector Store:** Qdrant / FAISS
- **Embeddings:** CLIP (multi-modal), text embedding model (semantic search)
- **Vision-Language Model:** GPT-4o / LLaVA
- **Observability:** Langfuse
- **Guardrails:** NeMo Guardrails

---

## Development Roadmap

| Week | AI Engineering | Full-Stack Integration |
|---|---|---|
| **1** | Multi-modal ingestion pipeline: PDF parsing, text chunking, image extraction, embedding into Qdrant | FastAPI scaffolding for async document upload & querying |
| **2** | LangGraph state machine + Supervisor node for query routing | Streamlit chat UI rendering agent thought process + referenced images |
| **Mid Review** | Reasoning audit — supervisor correctly chooses vector search vs. SQL execution | Vision check — VLM accurately extracts numerical data from bar charts |
| **3** | Self-RAG: agents detect irrelevant retrieval, rewrite queries, and retry | NeMo Guardrails integration to block out-of-scope answers |
| **4** | Langfuse integration for token usage, latency, and execution trace observability | UI polish + clickable citations linking to exact PDF page/chart |
| **Final Review** | Production-grade agentic system for autonomous reasoning over unstructured data | Hallucination-resistant enterprise search tool |

---

## Project Status

- ✅ Mid-Review demo passed
- ✅ Multiple ingestion/routing bugs fixed
- ⏳ Streamlit frontend ↔ agent endpoint wiring in progress
- ⏳ Self-RAG correction loop and guardrails integration pending

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/vedantdubey19/omnibrain.git
cd omnibrain

# Backend setup
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend setup
cd ../frontend
pip install -r requirements.txt
streamlit run app.py
```

### Environment Variables

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
DATABASE_URL=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

> ⚠️ Use `process.env` / environment-based config for all service URLs — never hardcode `localhost`, since it will break production and deployed builds.

---

## Example Query Flow

1. **Upload** a financial PDF via the FastAPI `/upload` endpoint
2. **Ask:** *"What was the YoY revenue growth shown in the Q3 chart, and how does it compare to the 5-year historical average?"*
3. **Supervisor routes** the query:
   - Vision Agent → extracts the Q3 chart's numeric values
   - SQL Agent → queries the historical revenue table
   - Search Agent → retrieves supporting narrative text from the filing
4. **Self-RAG loop** re-queries if any retrieval is irrelevant
5. **Guardrails** verify grounding before response is returned
6. **Response** is synthesized into a cited memo, each claim linked to its source page/chart

---

## License

MIT License — feel free to fork and build on this.

---

