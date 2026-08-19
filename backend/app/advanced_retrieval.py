from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from vector_store import query_similar_chunks
from embeddings import generate_embeddings

# Initialize Cross-Encoder model for precision reranking
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search_and_rerank(
    query: str, 
    corpus_chunks: List[Dict[str, Any]], 
    top_k: int = 2
) -> List[Dict[str, Any]]:
    """
    Combines BM25 Keyword Search and Vector Similarity Search, 
    then applies Cross-Encoder Reranking to return the most relevant contexts.
    """
    if not query or not corpus_chunks:
        return []
    
    # 1. Sparse Keyword Search (BM25)
    tokenized_corpus = [chunk["text"].lower().split() for chunk in corpus_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Sort chunks by BM25 relevance score
    bm25_results = [
        corpus_chunks[i] for i in sorted(range(len(bm25_scores)), key=lambda k: bm25_scores[k], reverse=True)
    ]
    
    # 2. Dense Vector Search (ChromaDB)
    query_chunk = [{"text": query, "chunk_id": "query"}]
    embedded_query = generate_embeddings(query_chunk)
    vector_results_raw = query_similar_chunks(embedded_query[0]["embedding"], n_results=len(corpus_chunks))
    
    retrieved_texts = vector_results_raw.get("documents", [[]])[0]
    vector_results = [c for c in corpus_chunks if c["text"] in retrieved_texts]
    
    # Merge candidate sets (Deduplicate)
    candidates = {c["text"]: c for c in (bm25_results + vector_results)}.values()
    candidate_list = list(candidates)
    
    # 3. Cross-Encoder Reranking
    pairs = [[query, chunk["text"]] for chunk in candidate_list]
    rerank_scores = reranker_model.predict(pairs)
    
    # Attach rerank scores and sort
    for idx, chunk in enumerate(candidate_list):
        chunk["rerank_score"] = float(rerank_scores[idx])
        
    reranked_chunks = sorted(candidate_list, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_chunks[:top_k]


if __name__ == "__main__":
    from chunking import chunk_text
    from embeddings import generate_embeddings
    from vector_store import store_embeddings
    
    # Sample Knowledge Base
    corpus_text = """
    Axelero Project 1 specializes in advanced RAG pipeline architectures.
    Dense vector search maps semantic meaning, but BM25 keyword matching captures exact model numbers or proper nouns.
    Cross-encoder reranking refines the combined retrieval candidates for sub-second precision.
    """
    
    chunks = chunk_text(corpus_text, chunk_size=80, chunk_overlap=15, metadata={"source": "advanced_rag_doc"})
    embedded_chunks = generate_embeddings(chunks)
    store_embeddings(embedded_chunks, collection_name="hybrid_collection")
    
    test_query = "How does BM25 keyword matching work with vector search?"
    final_top_results = hybrid_search_and_rerank(test_query, embedded_chunks, top_k=2)
    
    print(f"\nUser Query: '{test_query}'")
    print(f"Top Hybrid & Reranked Matches:")
    for idx, res in enumerate(final_top_results, 1):
        print(f"  [{idx}] (Score: {res['rerank_score']:.4f}) {res['text']}")