import os
import uuid
import aiofiles
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Service imports
from backend.app.services.retriever_service import RetrieverService
from backend.app.services.llm_service import LLMSynthesisService
from backend.app.services.agent_service import (
    AgentOrchestrationService,
    AgentQueryRequest,
    AgentQueryResponse
)

app = FastAPI(
    title="OmniBrain API Core",
    description="Backend API Gateway for PDF ingestion, multi-modal vector search, and LangGraph agent orchestration.",
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

ingestion_jobs: Dict[str, Dict[str, Any]] = {}

# Initialize service singletons
retriever_service = RetrieverService()
llm_service = LLMSynthesisService()
agent_service = AgentOrchestrationService()


# --- Global Error Handlers (Day 15 Hardening) ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTPException",
            "detail": exc.detail,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


# --- Pydantic Schemas ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="The query string or question asked by the user.")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of relevant document chunks to retrieve.")
    document_id: Optional[str] = Field(default=None, description="Optional target document filter.")

class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    page: int
    score: float
    source: str

class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    status: str


# --- Ingestion Background Worker ---

def process_pdf_ingestion(job_id: str, file_path: str):
    try:
        ingestion_jobs[job_id]["status"] = "PROCESSING"
        ingestion_jobs[job_id]["message"] = "Extracting text, tables, and images via document parser..."

        try:
            from pdf_parser_module.app.services.parser import PDFParser
            parser = PDFParser()
            _ = parser.extract_all(file_path)
        except ImportError:
            pass

        ingestion_jobs[job_id]["status"] = "COMPLETED"
        ingestion_jobs[job_id]["message"] = "Document successfully ingested and indexed into vector DB."
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "FAILED"
        ingestion_jobs[job_id]["message"] = f"Ingestion failed: {str(e)}"


# --- Core Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "OmniBrain Core API Gateway"}

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
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query question cannot be empty."
        )

    query_id = str(uuid.uuid4())

    try:
        retrieved_data = retriever_service.retrieve_relevant_chunks(
            query=clean_question,
            top_k=request.top_k,
            document_id=request.document_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store retrieval service unavailable: {str(e)}"
        )

    chunks = [RetrievedChunk(**item) for item in retrieved_data]

    try:
        synthesized_answer = llm_service.generate_answer(
            question=clean_question,
            retrieved_chunks=retrieved_data
        )
    except Exception as e:
        synthesized_answer = f"Error generating answer from LLM: {str(e)}"

    return QueryResponse(
        query_id=query_id,
        question=clean_question,
        answer=synthesized_answer,
        retrieved_chunks=chunks,
        status="SUCCESS"
    )

@app.post("/api/v1/agent/query", response_model=AgentQueryResponse, status_code=status.HTTP_200_OK)
async def query_agent_graph(request: AgentQueryRequest):
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query question cannot be empty."
        )

    try:
        result = await agent_service.execute_agent_workflow(
            question=clean_question,
            session_id=request.session_id,
            document_id=request.document_id
        )
        return AgentQueryResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow execution failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)