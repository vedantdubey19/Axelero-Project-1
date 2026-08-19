import os
import aiofiles

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import create_embeddings, search_similar


app = FastAPI(
    title="OmniBrain API Core",
    description="Backend API for PDF handling, text extraction, cleaning, embeddings and RAG service orchestration.",
    version="1.0.0"
)


# CORS configuration
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


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():

    return {
        "status": "healthy",
        "service": "OmniBrain Core API"
    }


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

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

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "saved_path": file_path,
        "size_bytes": file_size
    }


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

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

        pages = []
        full_text = ""

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            text = text.replace("\x00", "")

            pages.append({
                "page": page_number,
                "text": text
            })

            full_text += text + "\n\n"

        return {
            "filename": filename,
            "total_pages": len(reader.pages),
            "text": full_text,
            "pages": pages
        }

    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract PDF text: {str(e)}"
        )


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

@app.get("/api/v1/chunks/{filename}")
async def create_text_chunks(filename: str):

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

            full_text += text + "\n\n"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )

        chunks = splitter.split_text(full_text)

        return {
            "filename": filename,
            "total_pages": len(reader.pages),
            "total_chunks": len(chunks),
            "chunk_size": 800,
            "chunk_overlap": 120,
            "chunks": [
                {
                    "chunk_id": index + 1,
                    "text": chunk,
                    "characters": len(chunk)
                }
                for index, chunk in enumerate(chunks)
            ]
        }

    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create text chunks: {str(e)}"
        )


# ---------------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------------

@app.post("/api/v1/embed/{filename}")
async def embed_pdf(filename: str):

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
            detail=f"Embedding failed: {str(e)}"
        )


# ---------------------------------------------------------
# SEARCH / RETRIEVAL
# ---------------------------------------------------------

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

    if top_k < 1 or top_k > 10:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be between 1 and 10."
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


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )