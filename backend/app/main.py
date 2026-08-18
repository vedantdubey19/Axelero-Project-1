import os
import uuid
import aiofiles
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OmniBrain API Core",
    description="Backend API for PDF handling, Auth, and RAG service orchestration.",
    version="1.0.0"
)

# CORS setup for Streamlit frontend integration
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

# In-memory job status tracker
ingestion_jobs: Dict[str, Dict[str, Any]] = {}


# --- Pydantic Data Models ---

class QueryRequest(BaseModel):
    """Incoming user search/question payload."""
    question: str = Field(..., min_length=2, description="The query string or question asked by the user.")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of relevant document chunks to retrieve.")
    document_id: Optional[str] = Field(default=None, description="Optional target document filter.")


class RetrievedChunk(BaseModel):
    """Schema for context chunks returned by the retriever."""
    chunk_id: str
    content: str
    page: int
    score: float
    source: str


class QueryResponse(BaseModel):
    """Standardized API response for user queries."""
    query_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    status: str


# --- Ingestion Background Handler ---

def process_pdf_ingestion(job_id: str, file_path: str):
    try:
        ingestion_jobs[job_id]["status"] = "PROCESSING"
        ingestion_jobs[job_id]["message"] = "Extracting text, tables, and images..."

        # Downstream pipeline hooks (Humera & Saju)
        ingestion_jobs[job_id]["status"] = "COMPLETED"
        ingestion_jobs[job_id]["message"] = "Document successfully ingested and indexed into Qdrant."
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "FAILED"
        ingestion_jobs[job_id]["message"] = f"Ingestion failed: {str(e)}"


# --- Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "OmniBrain Core API"}


@app.post("/api/v1/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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

    job_id = str(uuid.uuid4())
    ingestion_jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "file_path": file_path,
        "status": "QUEUED",
        "message": "Ingestion job queued automatically after upload."
    }

    background_tasks.add_task(process_pdf_ingestion, job_id, file_path)

    return {
        "message": "File uploaded and ingestion pipeline triggered successfully",
        "job_id": job_id,
        "filename": file.filename,
        "saved_path": file_path,
        "size_bytes": file_size,
        "status": "QUEUED"
    }


@app.get("/api/v1/ingest/status/{job_id}", status_code=status.HTTP_200_OK)
async def get_ingestion_status(job_id: str):
    if job_id not in ingestion_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job with ID '{job_id}' not found."
        )
    return ingestion_jobs[job_id]


@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(request: QueryRequest):
    """
    Query endpoint that accepts user questions and prepares payload
    for Saju's retriever and LLM pipeline.
    """
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query question cannot be empty or whitespace."
        )

    query_id = str(uuid.uuid4())

    # Mock retrieval placeholder : Wired to Saju's Qdrant retriever
    mock_retrieved_chunks = [
        RetrievedChunk(
            chunk_id=str(uuid.uuid4()),
            content="Sample retrieved document text for context verification.",
            page=1,
            score=0.89,
            source="data/raw/sample.pdf"
        )
    ]

    return QueryResponse(
        query_id=query_id,
        question=clean_question,
        answer="This is a stub answer. Live LLM generation will be connected in Day 10.",
        retrieved_chunks=mock_retrieved_chunks,
        status="SUCCESS"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)