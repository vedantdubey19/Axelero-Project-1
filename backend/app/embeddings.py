import os
from typing import List, Dict, Any

# Option A: Sentence-Transformers (Free, local HuggingFace model)
# Install via: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    LOCAL_MODEL = None


def generate_embeddings(
    chunks: List[Dict[str, Any]], 
    model_provider: str = "local"
) -> List[Dict[str, Any]]:
    """
    Generates vector embeddings for a list of text chunk dictionaries.
    
    Args:
        chunks (List[Dict[str, Any]]): Output list from chunk_text function.
        model_provider (str): 'local' (SentenceTransformers) or 'openai'.
        
    Returns:
        List[Dict[str, Any]]: Chunks updated with 'embedding' vector field.
    """
    if not chunks:
        return []
    
    texts = [chunk["text"] for chunk in chunks]
    
    if model_provider == "local":
        if LOCAL_MODEL is None:
            raise RuntimeError("sentence-transformers is not installed. Run `pip install sentence-transformers`.")
        
        # Generate dense embeddings locally
        vectors = LOCAL_MODEL.encode(texts, show_progress_bar=False).tolist()
        
    elif model_provider == "openai":
        # Requires: pip install openai
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        vectors = [item.embedding for item in response.data]
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    
    # Attach embedding vectors back to chunk objects
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = vectors[i]
        chunk["vector_dim"] = len(vectors[i])
        
    return chunks


if __name__ == "__main__":
    from chunking import chunk_text
    
    sample_text = """
    Retrieval-Augmented Generation (RAG) combines semantic search with generative language models.
    Vector embeddings capture the semantic meaning of text in dense mathematical spaces.
    """
    
    # Step 1: Chunk text
    sample_chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    
    # Step 2: Generate Embeddings
    print("Generating local embeddings...")
    embedded_chunks = generate_embeddings(sample_chunks, model_provider="local")
    
    print(f"\nProcessed {len(embedded_chunks)} chunks successfully.")
    for c in embedded_chunks:
        print(f"Chunk ID: {c['chunk_id']} | Vector Dimension: {c['vector_dim']}")
        print(f"Embedding snippet: {c['embedding'][:5]}...\n")