import os
import aiofiles

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from .vector_store import create_embeddings, search_similar


app = FastAPI(
    title="OmniBrain API Core",
    description="Backend API for PDF handling, text extraction, cleaning, embeddings and RAG service orchestration.",
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


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "service": "OmniBrain Core API"
    }


@app.post("/api/v1/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):

    if (
        not file.filename.lower().endswith(".pdf")
        or file.content_type not in ALLOWED_MIME_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        file_size = 0

        async with aiofiles.open(file_path, "wb") as out_file:

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

    try:
        embedding_result = create_embeddings(file.filename)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create embeddings: {str(e)}"
        )

    return {
        "message": "File uploaded and embeddings created successfully",
        "filename": file.filename,
        "saved_path": file_path,
        "size_bytes": file_size,
        "embedding": embedding_result
    }


@app.get("/api/v1/extract/{filename}")
async def extract_pdf_text(filename: str):

    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found."
        )

    try:
        reader = PdfReader(file_path)

        full_text = ""

        for page in reader.pages:

            text = page.extract_text() or ""
            text = text.replace("\x00", "")

            full_text += text + "\n"

        return {
            "filename": filename,
            "text": full_text,
            "characters": len(full_text)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract PDF text: {str(e)}"
        )


@app.get("/api/v1/chunks/{filename}")
async def create_text_chunks(filename: str):

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found."
        )

    try:
        result = create_embeddings(filename)

        return {
            "message": "Embeddings created successfully",
            **result
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create embeddings: {str(e)}"
        )


@app.get("/api/v1/search")
async def search_documents(
    query: str,
    top_k: int = 3
):

    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    if top_k < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be at least 1."
        )

    try:
        results = search_similar(
            query=query,
            top_k=top_k
        )

        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/")
async def root():

    return {
        "message": "OmniBrain API is running",
        "docs": "/docs",
        "health": "/health"
    }