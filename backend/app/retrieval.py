from typing import List, Dict, Any
from embeddings import generate_embeddings
from vector_store import query_similar_chunks

def retrieve_context(query_text: str, top_k: int = 2) -> List[str]:
    """
    Unified retrieval pipeline:
    1. Converts raw user query string into an embedding vector.
    2. Queries ChromaDB for top-k similar document chunks.
    3. Returns retrieved text context strings.
    """
    if not query_text or not query_text.strip():
        return []
    
    # Wrap string in chunk format to pass to generate_embeddings
    query_chunk = [{"text": query_text, "chunk_id": "query"}]
    embedded_query = generate_embeddings(query_chunk)
    query_vector = embedded_query[0]["embedding"]
    
    # Retrieve top matches from ChromaDB
    search_results = query_similar_chunks(query_vector, n_results=top_k)
    
    # Extract list of document text snippets
    documents = search_results.get("documents", [[]])[0]
    return documents

if __name__ == "__main__":
    sample_query = "How do vector databases handle similarity?"
    retrieved_docs = retrieve_context(sample_query, top_k=2)
    
    print(f"\nUser Query: '{sample_query}'")
    print(f"Retrieved Context ({len(retrieved_docs)} chunks):")
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"  [{idx}] {doc}")