# OmniBrain — Setup & Deployment Guide

This guide covers local development (native and Docker Compose) and production cloud deployment (Render, Fly.io, and Qdrant Cloud).

---

## 1. Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose** (optional for local, required for containerized environments)
- **Git**
- Optional: OpenAI API Key (for GPT-4o-mini generation, query rewriting, and GPT-4o vision)
- Optional: Langfuse Account (for distributed LLM tracing and observability)
- Optional: Qdrant Cloud Account (free tier available at [cloud.qdrant.io](https://cloud.qdrant.io))

---

## 2. Local Development (Native Virtualenv)

### A. Clone and configure environment
```bash
git clone https://github.com/vedantdubey19/Axelero-Project-1.git
cd Axelero-Project-1
cp .env.example .env
```

### B. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### C. Run the Backend Gateway
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs & Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### D. Run the Streamlit Frontend
In a separate terminal:
```bash
source .venv/bin/activate
streamlit run app.py --server.port 8501
```
- Frontend UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## 3. Local Containerized Deployment (Docker Compose)

Launch the full stack (FastAPI backend, Streamlit frontend, and local Qdrant container) on a shared bridge network:

```bash
docker compose up --build
```

| Service | Container Name | Port | Description |
| :--- | :--- | :--- | :--- |
| **Backend** | `omnibrain-backend` | `8000` | FastAPI gateway with LangGraph orchestration |
| **Frontend** | `omnibrain-frontend` | `8501` | Streamlit multimodal interactive dashboard |
| **Qdrant** | `omnibrain-qdrant` | `6333` | Vector database with persistence volume |

### Verification Endpoints:
- Backend Health: `http://localhost:8000/health`
- Swagger OpenAPI: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`
- Qdrant Dashboard: `http://localhost:6333/dashboard`

---

## 4. Managed Vector Database: Qdrant Cloud Setup

For production cloud deployments without running a persistent Docker container for storage:
1. Register for a free account at [cloud.qdrant.io](https://cloud.qdrant.io).
2. Create a free 1GB cluster (e.g. on AWS or GCP).
3. Under cluster settings, copy your **Cluster URL** (e.g. `https://xxxxxxxx.us-east4-0.gcp.cloud.qdrant.io:6333`) and generate an **API Key**.
4. Set these in your environment or cloud dashboard:
   ```bash
   QDRANT_URL=https://xxxxxxxx.us-east4-0.gcp.cloud.qdrant.io:6333
   QDRANT_API_KEY=your-secret-api-key
   ```
5. OmniBrain's `RetrieverService` detects `QDRANT_URL` and automatically routes index and search requests to the managed cloud cluster.

---

## 5. Production Cloud Deployment

### Option 1: Render (`render.yaml`)
OmniBrain includes an infrastructure-as-code specification file (`render.yaml`) for deploying the backend and frontend services:

1. Link your GitHub repository to your Render account.
2. Select **Blueprints** -> **New Blueprint Instance**.
3. Render automatically discovers `render.yaml` and provisions:
   - `omnibrain-backend`: Web service built from `./Dockerfile`, listening on port `8000`.
   - `omnibrain-frontend`: Web service built from `./frontend/Dockerfile`, listening on port `8501`.
4. In the Render Dashboard under **Environment Variables**, set:
   - `OPENAI_API_KEY`: Your OpenAI key.
   - `QDRANT_URL` and `QDRANT_API_KEY`: Your Qdrant Cloud credentials.
   - `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY`: (Optional) For distributed tracing.
5. In the frontend service settings, verify `BACKEND_URL` points to the public URL or internal host of `omnibrain-backend`.

### Option 2: Fly.io (`fly.toml`)
To deploy the backend to Fly.io:
```bash
fly launch --config fly.toml
fly secrets set OPENAI_API_KEY="sk-..." QDRANT_URL="https://..." QDRANT_API_KEY="..."
fly deploy
```

---

## 6. Cross-Origin Resource Sharing (CORS) Configuration

When frontend and backend are hosted on distinct domains (e.g., frontend on `https://omnibrain.streamlit.app` and backend on `https://omnibrain-backend.onrender.com`):
- Set `ALLOWED_ORIGINS` on the backend service to allow your frontend domain:
  ```bash
  ALLOWED_ORIGINS=https://omnibrain.streamlit.app,http://localhost:8501
  ```
- By default, `ALLOWED_ORIGINS=*` is configured for seamless deployment across preview environments.

---

## 7. Running the Automated Test Suite

OmniBrain maintains a comprehensive test suite across unit, integration, and end-to-end supervisor routing layers:

```bash
# Run all core tests
pytest tests/ -v

# Run PDF parser unit tests
pytest pdf_parser_module/tests/ -v

# Run full test suite with coverage summary
pytest -v
```

