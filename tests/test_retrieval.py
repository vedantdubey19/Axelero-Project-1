import sys
import os
import uuid
from qdrant_client.models import PointStruct, VectorParams, Distance

# Add backend/app to Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app')))

try:
    from backend.app.services.retriever_service import RetrieverService
    from backend.app.chunking import chunk_text
except ImportError:
    from services.retriever_service import RetrieverService
    from chunking import chunk_text


def test_canonical_qdrant_retrieval_pipeline():
    """
    Tests the canonical Qdrant-backed RetrieverService pipeline.
    Validates chunking, embedding generation, Qdrant indexing, and similarity retrieval.
    """
    # 1. Chunk sample document text
    sample_text = (
        "Dense vector embeddings store high-dimensional text representations for fast retrieval. "
        "OmniBrain uses Qdrant vector database to perform semantic similarity search."
    )
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=10, metadata={"source": "test_doc.pdf"})
    assert len(chunks) > 0, "Chunking failed to generate chunks."

    # 2. Instantiate RetrieverService with in-memory test collection
    collection_name = "test_canonical_chunks"
    retriever = RetrieverService(collection_name=collection_name)

    # 3. Embed text chunks using retriever embedder
    texts = [c["text"] for c in chunks]
    embeddings = retriever.embedder.encode(texts, convert_to_numpy=True).tolist()
    assert len(embeddings) == len(chunks), "Embeddings count mismatch."
    assert len(embeddings[0]) == 384, "Embedding vector dimension mismatch."

    # 4. Upsert into Qdrant collection
    vector_size = len(embeddings[0])
    try:
        retriever.client.get_collection(collection_name)
    except Exception:
        retriever.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    points = []
    for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"test_{i}"))
        points.append(PointStruct(
            id=point_id,
            vector=emb,
            payload={
                "content": chunk["text"],
                "source": "test_doc.pdf",
                "page": 1
            }
        ))

    retriever.client.upsert(collection_name=collection_name, points=points)

    # 5. Execute retrieval
    results = retriever.retrieve_relevant_chunks("What do vector embeddings store?", top_k=2)
    assert len(results) > 0, "RetrieverService returned no matching chunks."
    assert "content" in results[0], "Missing content in retrieved chunk."
    assert results[0]["source"] == "test_doc.pdf", "Source metadata mismatch."
    assert results[0]["score"] > 0, "Invalid similarity score."


if __name__ == "__main__":
    test_canonical_qdrant_retrieval_pipeline()