import os
import sys
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import operator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END


class SelfRAGState(TypedDict):
    query: str
    documents: List[Dict[str, Any]]
    generation: str
    is_relevant: bool
    retry_count: int


def retrieve_with_citations_node(state: SelfRAGState) -> SelfRAGState:
    """Retrieves context chunks attached with explicit citation metadata."""
    query = state["query"]
    print(f"\n[Retriever] Searching context for: '{query}'")

    # Simulated retrieval with citation metadata attached
    mock_docs = [
        {
            "content": "Hybrid search combines BM25 keyword matching with dense vector embeddings for optimal RAG precision.",
            "source": "Axelero_RAG_Docs.pdf",
            "page": 4,
            "chunk_id": "chunk_102"
        },
        {
            "content": "LangGraph manages stateful multi-agent execution using cyclic graphs and conditional routing nodes.",
            "source": "LangGraph_Architecture.pdf",
            "page": 12,
            "chunk_id": "chunk_305"
        }
    ]
    
    state["documents"] = mock_docs
    return state


def grade_documents_node(state: SelfRAGState) -> SelfRAGState:
    """Self-RAG Evaluator: Grades retrieved document relevance against the user query."""
    query = state["query"].lower()
    docs = state["documents"]
    
    # Keyword relevance check (simulating LLM relevance grading)
    relevant = any(
        any(term in doc["content"].lower() for term in query.split()) 
        for doc in docs
    )
    
    state["is_relevant"] = relevant
    print(f"[Self-RAG Grader] Context Relevance Score: {'PASSED' if relevant else 'FAILED'}")
    return state


def generate_cited_answer_node(state: SelfRAGState) -> SelfRAGState:
    """Generates an answer strictly backed by cited sources."""
    docs = state["documents"]
    context_str = ""
    citations = []
    
    for idx, doc in enumerate(docs, 1):
        context_str += f"\n[{idx}] {doc['content']}"
        citations.append(f"[{idx}] Source: {doc['source']} (Page {doc['page']})")
    
    response = (
        f"Based on retrieved sources:\n{context_str}\n\n"
        f"**Citations:**\n" + "\n".join(citations)
    )
    
    state["generation"] = response
    return state


def fallback_query_rewrite_node(state: SelfRAGState) -> SelfRAGState:
    """Fallback Loop: Rewrites the query if initial retrieval failed relevance grading."""
    state["retry_count"] += 1
    new_query = f"Optimized search query: {state['query']} (Attempt {state['retry_count']})"
    print(f"[Self-RAG Rewriter] Low relevance detected. Rewriting query -> '{new_query}'")
    state["query"] = new_query
    return state


def check_relevance_router(state: SelfRAGState) -> str:
    """Routing decision based on relevance check and retry threshold."""
    if state["is_relevant"] or state["retry_count"] >= 2:
        return "generate"
    return "rewrite"


# Construct Self-RAG Graph Workflow
workflow = StateGraph(SelfRAGState)

workflow.add_node("retrieve", retrieve_with_citations_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("generate", generate_cited_answer_node)
workflow.add_node("rewrite", fallback_query_rewrite_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    check_relevance_router,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

self_rag_app = workflow.compile()


if __name__ == "__main__":
    print("--- Running Self-RAG Execution Test ---")
    initial_state = {
        "query": "How does hybrid search work in RAG?",
        "documents": [],
        "generation": "",
        "is_relevant": False,
        "retry_count": 0
    }
    
    result = self_rag_app.invoke(initial_state)
    print("\n--- Final Pipeline Output ---")
    print(result["generation"])