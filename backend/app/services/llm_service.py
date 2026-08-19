import os
from typing import List, Dict, Any

class LLMSynthesisService:
    """
    Synthesizes final RAG answers using retrieved document context.
    Supports OpenAI, Gemini, or local fallback.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", timeout_seconds: float = 15.0):
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats context chunks and sends a prompt to the LLM.
        """
        if not retrieved_chunks:
            return "I could not find any relevant information in the uploaded documents to answer your question."

        # Build context block
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            page = chunk.get("page", "N/A")
            source = chunk.get("source", "document")
            content = chunk.get("content", "").strip()
            context_str += f"\n[Context {i} | Source: {source}, Page: {page}]\n{content}\n"

        prompt = (
            "You are OmniBrain, an expert multi-modal enterprise AI assistant. "
            "Answer the user's question accurately using ONLY the context provided below. "
            "If the answer cannot be determined from the context, state that clearly. "
            "Cite the context sources where relevant.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 1. Live OpenAI Integration (if API key available)
        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that answers based on provided documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error during LLM generation: {str(e)}"

        # 2. Local Fallback Synthesis (when running offline without API keys)
        return (
            f"[Offline Synthesis] Based on {len(retrieved_chunks)} retrieved context passage(s):\n"
            f"- Primary reference (Page {retrieved_chunks[0].get('page', 1)}): "
            f"\"{retrieved_chunks[0].get('content', '')[:200]}...\""
        )