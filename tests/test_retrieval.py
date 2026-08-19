import sys
import os

# Add backend/app to Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app')))

from chunking import chunk_text
from embeddings import generate_embeddings
from vector_store import store_embeddings
from retrieval import retrieve_context

def test_full_rag_pipeline():
    # 1. Test Text
    sample_text = "Dense vector embeddings store high-dimensional text representations for fast retrieval."
    
    # 2. Chunking
    chunks = chunk_text(sample_text, chunk_size=40, chunk_overlap=10, metadata={"source": "test_doc"})
    assert len(chunks) > 0, "Chunking failed to generate chunks."
    
    # 3. Embeddings
    embedded_chunks = generate_embeddings(chunks)
    assert "embedding" in embedded_chunks[0], "Embedding field missing."
    assert len(embedded_chunks[0]["embedding"]) == 384, "Embedding vector dimension mismatch."
    
    # 4. Storage
    store_embeddings(embedded_chunks, collection_name="test_collection")
    
    # 5. Retrieval
    results = retrieve_context("What do vector embeddings store?", top_k=1)
    assert len(results) > 0, "Retrieval returned no context."
    print("\n✅ All RAG pipeline tests passed successfully!")

if __name__ == "__main__":
    test_full_rag_pipeline()