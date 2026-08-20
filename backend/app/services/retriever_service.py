import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class RetrieverService:
    """
    Service responsible for embedding queries and searching relevant chunks from Qdrant.
    """

    def __init__(
        self,
        collection_name: str = "omnibrain_text_chunks",
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        self.collection_name = collection_name
        qdrant_host = host or os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = port or int(os.getenv("QDRANT_PORT", "6333"))

        # Fallback to local in-memory storage if local Qdrant container is not running
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=2.0)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(location=":memory:")

        # Lazy-loaded embedding model
        self._embedder = None

    @property
    def embedder(self):
        """Lazy load SentenceTransformer embedder on first access."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 3,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Converts text query to vector and retrieves top-k matching points from Qdrant.
        """
        query_vector = self.embedder.encode(query, convert_to_numpy=True).tolist()

        # Optional payload filtering by specific document_id
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter
                )
                search_results = response.points
            elif hasattr(self.client, "search"):
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter
                )
            else:
                search_results = []
        except Exception:
            # Safe fallback if collection is not yet populated
            return []

        formatted_chunks = []
        for point in search_results:
            formatted_chunks.append({
                "chunk_id": str(point.id),
                "content": point.payload.get("content", ""),
                "page": point.payload.get("page", 1),
                "score": float(point.score),
                "source": point.payload.get("source", "unknown")
            })

        return formatted_chunks
