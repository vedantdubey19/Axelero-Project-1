# ---- Backend Dockerfile (FastAPI service) ----
# Multi-stage build: smaller final image, faster rebuilds via layer caching.

FROM python:3.11-slim AS base

# Prevent .pyc files & enable unbuffered stdout (useful for docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by pdfplumber / PyMuPDF / pytesseract (OCR) at runtime.
# Remove tesseract-ocr / poppler-utils lines below if backend doesn't need them directly
# (Humera's parser module may run in the same container).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Dependency layer (cached separately so code changes don't reinstall deps) ---
# NOTE: adjust path if requirements.txt lives elsewhere (e.g. backend/requirements.txt).
# If backend and pdf_parser_module have SEPARATE requirement files, merge them into
# one requirements.txt at repo root, OR copy+install both here.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- App code layer ---
COPY . .

EXPOSE 8000

# Adjust module path if your FastAPI entrypoint differs from backend/app/main.py
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
