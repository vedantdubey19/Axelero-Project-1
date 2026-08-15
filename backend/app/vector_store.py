import chromadb
from typing import List, Dict, Any

# Initialize local persistent client
client = chromadb.PersistentClient(path="./chroma_db")

def store_embeddings(chunks: List[Dict[str, Any]], collection_name: str = "rag_collection"):
    """
    Stores vector embeddings and metadata in ChromaDB.
    """
    collection = client.get_or_create_collection(name=collection_name)
    
    ids = [f"chunk_{c['chunk_id']}" for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    
    # Process metadata: pass None if empty, or ensure non-empty dicts
    metadatas = []
    for c in chunks:
        meta = c.get("metadata", {})
        # If empty dict, provide default key or use None
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

if __name__ == "__main__":
    from chunking import chunk_text
    from embeddings import generate_embeddings
    
    text = "Vector databases allow fast similarity search over high-dimensional embeddings."
    
    # Generate chunks with default metadata
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10, metadata={"source": "sample_text.txt"})
    embedded_chunks = generate_embeddings(chunks)
    
    # Store in ChromaDB
    store_embeddings(embedded_chunks)
    
    # Query test
    query_vec = embedded_chunks[0]["embedding"]
    matches = query_similar_chunks(query_vec, n_results=1)
    print("\nTop Query Result:")
    print(matches["documents"][0])