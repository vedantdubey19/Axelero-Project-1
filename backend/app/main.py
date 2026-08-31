import os
import re
import uuid
import aiofiles
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Service imports with robust fallbacks
try:
    from backend.app.services.guardrails_service import GuardrailsService
    from backend.app.services.retriever_service import RetrieverService
    from backend.app.services.llm_service import LLMSynthesisService
    from backend.app.services.citation_service import CitationService
    from backend.app.services.agent_service import (
        AgentOrchestrationService,
        AgentQueryRequest,
        AgentQueryResponse
    )
except ImportError:
    from services.guardrails_service import GuardrailsService
    from services.retriever_service import RetrieverService
    from services.llm_service import LLMSynthesisService
    from services.citation_service import CitationService
    from services.agent_service import (
        AgentOrchestrationService,
        AgentQueryRequest,
        AgentQueryResponse
    )

app = FastAPI(
    title="OmniBrain API Core",
    description="Backend API Gateway for PDF ingestion, multi-modal vector search, and LangGraph agent orchestration.",
    version="1.0.0"
)

guardrails_service = GuardrailsService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.abspath("data/raw")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 25
ALLOWED_MIME_TYPES = ["application/pdf"]
SIMILARITY_CONFIDENCE_THRESHOLD = 0.65

ingestion_jobs: Dict[str, Dict[str, Any]] = {}

# Initialize service singletons
retriever_service = RetrieverService()
llm_service = LLMSynthesisService()
citation_service = CitationService(raw_docs_dir=UPLOAD_DIR)
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
    original_question: Optional[str] = None
    rewritten_query: Optional[str] = None
    retried: bool = False
    low_confidence: bool = False
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    status: str

class CitationResponse(BaseModel):
    filename: str
    page_number: int
    total_pages: int
    snippet: str
    char_count: int


# --- Ingestion Background Worker ---

def process_pdf_ingestion(job_id: str, file_path: str):
    try:
        ingestion_jobs[job_id]["status"] = "PROCESSING"
        ingestion_jobs[job_id]["message"] = "Extracting text, tables, and images via document parser..."

        parse_result = None
        try:
            from pdf_parser_module.app.services.parser import parse_pdf
            from pathlib import Path
            parse_result = parse_pdf(job_id, Path(file_path), os.path.basename(file_path))
        except Exception as parse_err:
            ingestion_jobs[job_id]["status"] = "FAILED"
            ingestion_jobs[job_id]["message"] = f"PDF parsing failed: {str(parse_err)}"
            return

        # Index parsed text chunks into Qdrant for retriever_service
        filename = os.path.basename(file_path)
        try:
            _index_parsed_text_to_qdrant(parse_result, filename)
        except Exception as index_err:
            ingestion_jobs[job_id]["status"] = "FAILED"
            ingestion_jobs[job_id]["message"] = f"Vector indexing failed: {str(index_err)}"
            return

        ingestion_jobs[job_id]["status"] = "COMPLETED"
        ingestion_jobs[job_id]["message"] = "Document successfully ingested and indexed into vector DB."
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "FAILED"
        ingestion_jobs[job_id]["message"] = f"Ingestion failed: {str(e)}"


def _index_parsed_text_to_qdrant(parse_result, filename: str):
    """
    Indexes parsed PDF text into the Qdrant collection used by RetrieverService.
    Splits each page's text into chunks and upserts with embeddings.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from qdrant_client.models import PointStruct, VectorParams, Distance

    if parse_result is None:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    points = []
    all_texts = []
    all_metadata = []

    for page in parse_result.pages:
        text = page.text.strip() if page.text else ""
        if not text:
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadata.append({
                "source": filename,
                "page": page.page_number,
                "content": chunk
            })

    if not all_texts:
        return

    # Use the same embedder as retriever_service
    embeddings = retriever_service.embedder.encode(all_texts, convert_to_numpy=True).tolist()

    collection_name = retriever_service.collection_name
    vector_size = len(embeddings[0])

    # Ensure collection exists
    try:
        retriever_service.client.get_collection(collection_name)
    except Exception:
        retriever_service.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    for i, (embedding, meta) in enumerate(zip(embeddings, all_metadata)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{meta['page']}_{i}"))
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload=meta
        ))

    retriever_service.client.upsert(
        collection_name=collection_name,
        points=points
    )


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

@app.get("/api/v1/search", status_code=status.HTTP_200_OK)
async def search_documents(query: str, top_k: int = 3, document_id: Optional[str] = None):
    """
    Search endpoint compatible with Streamlit frontend.
    """
    clean_query = query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty."
        )

    try:
        retrieved_data = retriever_service.retrieve_relevant_chunks(
            query=clean_query,
            top_k=top_k,
            document_id=document_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store retrieval service unavailable: {str(e)}"
        )

    matches = []
    for chunk in retrieved_data:
        matches.append({
            "text": chunk.get("content", ""),
            "metadata": {
                "filename": chunk.get("source", "document.pdf"),
                "chunk_id": chunk.get("chunk_id", ""),
                "page": chunk.get("page", 1)
            },
            "distance": round(max(0.0, 1.0 - float(chunk.get("score", 0.0))), 4)
        })

    return {"query": clean_query, "matches": matches, "count": len(matches)}

@app.get("/api/v1/citations/{filename}/{page_number}", response_model=CitationResponse, status_code=status.HTTP_200_OK)
async def get_citation_page(filename: str, page_number: int):
    """Serves exact PDF page snippets for frontend citation popups."""
    active_service = CitationService(raw_docs_dir=os.path.abspath("data/raw"))
    result = active_service.get_page_snippet(filename, page_number)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Citation page {page_number} for file '{filename}' not found."
        )
    return CitationResponse(**result)

@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(request: QueryRequest):
    """Self-RAG query execution with confidence evaluation and automatic rewrite loop."""
    clean_question = request.question.strip()
    if not clean_question or not re.sub(r'[\W_]+', '', clean_question).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query question cannot be empty or solely punctuation."
        )

    query_id = str(uuid.uuid4())
    rewritten_query_str = None
    has_retried = False
    is_safe, rejection_message = guardrails_service.validate_input(clean_question)
    if not is_safe:
        return QueryResponse(
            query_id=query_id,
            question=clean_question,
            answer=rejection_message,
            retrieved_chunks=[],
            status="BLOCKED_BY_GUARDRAILS"
        )
    # 1. Initial Retrieval Pass
    try:
        retrieved_data = retriever_service.retrieve_relevant_chunks(
            query=clean_question,
            top_k=request.top_k,
            document_id=request.document_id
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Retrieval error: {str(e)}")

    # 2. Self-RAG Evaluation Loop: Trigger retry if empty or average score < threshold
    avg_score = (
        sum(item.get("score", 0.0) for item in retrieved_data) / len(retrieved_data)
        if retrieved_data else 0.0
    )
    is_low_confidence = (not retrieved_data) or (avg_score < SIMILARITY_CONFIDENCE_THRESHOLD)

    if is_low_confidence:
        has_retried = True
        rewritten_query_str = llm_service.rewrite_query(clean_question)
        retry_data = retriever_service.retrieve_relevant_chunks(
            query=rewritten_query_str,
            top_k=request.top_k,
            document_id=request.document_id
        )
        if retry_data:
            retrieved_data = retry_data
            retry_avg_score = (
                sum(item.get("score", 0.0) for item in retry_data) / len(retry_data)
            )
            is_low_confidence = retry_avg_score < SIMILARITY_CONFIDENCE_THRESHOLD
        else:
            is_low_confidence = True

    chunks = [RetrievedChunk(**item) for item in retrieved_data]

    # 3. LLM Synthesis
    try:
        synthesized_answer = llm_service.generate_answer(
            question=rewritten_query_str or clean_question,
            retrieved_chunks=retrieved_data
        )
    except Exception as e:
        synthesized_answer = f"Error during generation: {str(e)}"

    return QueryResponse(
        query_id=query_id,
        question=clean_question,
        original_question=clean_question if has_retried else None,
        rewritten_query=rewritten_query_str,
        retried=has_retried,
        low_confidence=is_low_confidence,
        answer=synthesized_answer,
        retrieved_chunks=chunks,
        status="SUCCESS"
    )

@app.post("/api/v1/agent/query", response_model=AgentQueryResponse, status_code=status.HTTP_200_OK)
async def query_agent_graph(request: AgentQueryRequest):
    clean_question = request.question.strip()
    if not clean_question or not re.sub(r'[\W_]+', '', clean_question).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query question cannot be empty or solely punctuation."
        )
        # Guardrails Input Rail Check (Days 22-23)
    is_safe, rejection_message = guardrails_service.validate_input(clean_question)
    if not is_safe:
        return AgentQueryResponse(
            query_id=str(uuid.uuid4()),
            session_id=request.session_id,
            question=clean_question,
            routed_agent="GuardrailsAgent",
            final_answer=rejection_message,
            execution_steps=[],
            referenced_sources=[],
            status="BLOCKED_BY_GUARDRAILS"
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