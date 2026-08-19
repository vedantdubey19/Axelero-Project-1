# OmniBrain — Local Setup Guide

## Prerequisites
- Docker Desktop (or Docker Engine + Compose plugin) installed
- Git
- Python 3.11 (only needed if running services outside Docker for debugging)

## 1. Clone & configure environment
```bash
git clone https://github.com/vedantdubey19/Axelero-Project-1.git
cd Axelero-Project-1
cp .env.example .env
# edit .env and add your real LLM API key etc.
```

## 2. Build and start all services
```bash
docker compose up --build
```

This starts three containers on a shared bridge network (`omnibrain-net`):

| Service   | Container name       | Port  | Purpose                        |
|-----------|-----------------------|-------|---------------------------------|
| backend   | omnibrain-backend     | 8000  | FastAPI (upload/query/ingest)  |
| frontend  | omnibrain-frontend    | 8501  | Streamlit UI                   |
| qdrant    | omnibrain-qdrant      | 6333  | Vector database                |

## 3. Verify it's working
- Backend health / docs: http://localhost:8000/docs (FastAPI auto Swagger UI)
- Frontend: http://localhost:8501
- Qdrant dashboard: http://localhost:6333/dashboard

## 4. Common commands
```bash
docker compose up -d          # run in background
docker compose logs -f backend  # tail backend logs
docker compose down           # stop everything
docker compose down -v        # stop and wipe volumes (fresh Qdrant/data)
```

## 5. Running tests locally (outside Docker)
```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest -v
```

## 6. Known gaps (see Repository Review Report, 18 Aug)
- `process_pdf_ingestion()` still uses mock stubs — needs to call Humera's parser + Saju's vector DB.
- `/api/v1/query` returns a hardcoded placeholder — needs live LLM wiring (see `.env.example` → `LLM_PROVIDER`).
- Frontend is a standalone script, not yet calling the backend API.
