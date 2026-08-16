import os
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OmniBrain API Core",
    description="Backend API for PDF handling, Auth, and RAG service orchestration.",
    version="1.0.0"
)

# Setup to allow request routing from Streamlit frontend
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
    """Health check route to verify backend service status."""
    return {"status": "healthy", "service": "OmniBrain Core API"}


@app.post("/api/v1/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Day 3-5 Milestone: Async PDF upload endpoint with validation.
    - Validates file extension and MIME type.
    - Uses non-blocking async disk writes (aiofiles).
    - Enforces file size limitations.
    """
    # 1. Extension & MIME Type Validation (Day 4 Requirement)
    if not file.filename.endswith(".pdf") or file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are supported."
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # 2. Non-blocking Async File Storage (Day 5 Requirement)
    try:
        file_size = 0
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
                file_size += len(chunk)

                # Size validation limit check
                if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    # Clean up partial file if limit exceeded
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)