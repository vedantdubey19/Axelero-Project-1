import os
from typing import List, Dict, Any

try:
    from backend.app.services.tracing_service import tracing_service
except ImportError:
    try:
        from services.tracing_service import tracing_service
    except ImportError:
        class _DummyTracing:
            def observe(self, *a, **k):
                def d(f):
                    return f
                return d
        tracing_service = _DummyTracing()


class LLMSynthesisService:
    """
    Synthesizes final RAG answers and provides query-rewriting for Self-RAG loops.
    Instrumented with Langfuse observability for token, latency, and generation tracing.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", timeout_seconds: float = 15.0):
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    @tracing_service.observe(name="self_rag_rewrite_query", as_type="generation")
    def rewrite_query(self, vague_query: str) -> str:
        """
        Self-RAG Rewriter: Expands vague queries into searchable domain terms.
        """
        if not self.api_key:
            return f"{vague_query} detailed summary and key points"

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query optimizer. Given a user query, rewrite it to be specific and keyword-rich for dense vector retrieval. Return only the rewritten query string."
                    },
                    {"role": "user", "content": vague_query}
                ],
                temperature=0.1,
                max_tokens=60
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return f"{vague_query} detailed summary and key points"

    @tracing_service.observe(name="llm_generate_answer", as_type="generation")
    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return "I could not find any relevant information in the uploaded documents to answer your question."

        context_str = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            page = chunk.get("page", "N/A")
            source = os.path.basename(chunk.get("source", "document"))
            content = chunk.get("content", "").strip()
            context_str += f"\n[Context {i} | Source: {source}, Page: {page}]\n{content}\n"

        prompt = (
            "You are OmniBrain, an enterprise multi-modal AI assistant. "
            "Answer the question accurately using ONLY the context provided below. "
            "Cite the source and page number for every key claim.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a strictly grounded AI assistant answering from supplied documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error during LLM generation: {str(e)}"

        return (
            f"[Offline Synthesis] Retrieved {len(retrieved_chunks)} passage(s). "
            f"Primary Reference (Page {retrieved_chunks[0].get('page', 1)}): "
            f"\"{retrieved_chunks[0].get('content', '')[:200]}...\""
        )