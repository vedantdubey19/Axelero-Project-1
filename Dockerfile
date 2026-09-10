FROM python:3.11-slim

# Prevent .pyc files & enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (including Tesseract OCR & poppler for PDF parsing fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install CPU-only torch first — avoids downloading the ~2GB of CUDA/GPU
# libraries that the default torch wheel pulls in, which caused a timeout
# on slower connections
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else, with a longer timeout and retries for slow connections
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt

# Copy project files
COPY . .

# Expose backend API port
EXPOSE 8000

# Start FastAPI backend
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]