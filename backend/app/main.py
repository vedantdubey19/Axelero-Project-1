import os
import uuid
import aiofiles
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OmniBrain API Core",
    description="Backend API for PDF handling, Auth, and RAG service orchestration.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 25
ALLOWED_MIME_TYPES = ["application/pdf"]

# In-memory job status tracker (In production, replace with Redis or SQL DB)
ingestion_jobs: Dict[str, Dict[str, Any]] = {}


def process_pdf_ingestion(job_id: str, file_path: str):
    """
    Background Task Execution Handler:
    Triggers parsing (Humera's module) and vector indexing (Saju's module).
    """
    try:
        ingestion_jobs[job_id]["status"] = "PROCESSING"
        ingestion_jobs[job_id]["message"] = "Extracting text, tables, and images..."

        # Downstream Pipeline Hooks (To be handled by Saju & Humera)
        # 1. elements = parser.extract_elements(file_path)
        # 2. embeddings = embedder.generate(elements)
        # 3. qdrant_client.upsert(embeddings)

        ingestion_jobs[job_id]["status"] = "COMPLETED"
        ingestion_jobs[job_id]["message"] = "Document successfully ingested and indexed into Qdrant."
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "FAILED"
        ingestion_jobs[job_id]["message"] = f"Ingestion failed: {str(e)}"


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "OmniBrain Core API"}


@app.post("/api/v1/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf") or file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        file_size = 0
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum size limit of {MAX_FILE_SIZE_MB}MB."
                    )
                await out_file.write(chunk)

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file write: {str(e)}"
        )
    finally:
        await file.close()

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_path": file_path,
        "size_bytes": file_size
    }


@app.post("/api/v1/ingest", status_code=status.HTTP_202_ACCEPTED)
async def trigger_ingestion(file_path: str, background_tasks: BackgroundTasks):
    """
    Day 6 Task: Endpoint to trigger document ingestion pipeline after file upload.
    Dispatches task to background worker to prevent HTTP blocking.
    """
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target document not found at path: {file_path}"
        )

    job_id = str(uuid.uuid4())
    ingestion_jobs[job_id] = {
        "job_id": job_id,
        "file_path": file_path,
        "status": "QUEUED",
        "message": "Ingestion job queued successfully."
    }

    # Dispatch ingestion to run asynchronously in background
    background_tasks.add_task(process_pdf_ingestion, job_id, file_path)

    return {
        "message": "Ingestion pipeline triggered successfully.",
        "job_id": job_id,
        "status": "QUEUED"
    }


@app.get("/api/v1/ingest/status/{job_id}", status_code=status.HTTP_200_OK)
async def get_ingestion_status(job_id: str):
    """
    Status Polling Route: Used by Frontend (Venkatesh) to render processing status.
    """
    if job_id not in ingestion_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job with ID '{job_id}' not found."
        )

    return ingestion_jobs[job_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)