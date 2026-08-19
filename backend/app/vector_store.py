import os
import chromadb
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = "data/raw"
CHROMA_DIR = "./data/chroma_db"

os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize local persistent client
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Shared embedding model instance
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# --- Saju's RAG Pipeline Interface ---

def store_embeddings(chunks: List[Dict[str, Any]], collection_name: str = "rag_collection"):
    """
    Stores vector embeddings and metadata in ChromaDB.
    """
    collection = client.get_or_create_collection(name=collection_name)
    
    ids = [f"chunk_{c['chunk_id']}" for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    
    metadatas = []
    for c in chunks:
        meta = c.get("metadata", {})
        metadatas.append(meta if meta else {"source": "default_doc"})
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully stored {len(chunks)} chunks in collection '{collection_name}'.")


def query_similar_chunks(query_embedding: List[float], n_results: int = 1, collection_name: str = "rag_collection"):
    """
    Retrieves the top-k most similar text chunks based on query vector.
    """
    collection = client.get_or_create_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results


# --- Venkatesh's Document Search Interface ---

def create_embeddings(filename: str, collection_name: str = "omnibrain_documents"):
    """
    Processes uploaded PDF from data/raw, splits text, generates embeddings, and upserts into Chroma.
    """
    file_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {filename}")

    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text.replace("\x00", "") + "\n"
    except Exception:
        # Fallback to PyMuPDF if pypdf is unavailable
        import fitz
        doc = fitz.open(file_path)
        full_text = "".join(page.get_text() for page in doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(full_text)

    if not chunks:
        raise ValueError("No text found in PDF.")

    embedder = get_embedder()
    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    collection = client.get_or_create_collection(name=collection_name)

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"filename": filename, "chunk_id": i + 1} for i in range(len(chunks))]
    )

    return {
        "filename": filename,
        "total_chunks": len(chunks),
        "embedding_dimension": len(embeddings[0]),
        "collection": collection_name
    }


def search_similar(query: str, top_k: int = 3, collection_name: str = "omnibrain_documents"):
    """
    Searches ChromaDB for matching passages and returns ranked matches.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    collection = client.get_or_create_collection(name=collection_name)
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    matches = []
    for i, document in enumerate(documents):
        matches.append({
            "rank": i + 1,
            "text": document,
            "distance": distances[i] if i < len(distances) else None,
            "metadata": metadatas[i] if i < len(metadatas) else {}
        })

    return {"query": query, "top_k": top_k, "matches": matches}
