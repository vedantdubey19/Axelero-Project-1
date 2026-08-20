import os
import sys
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

# Support package and standalone paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.app.services.retriever_service import RetrieverService
    from backend.app.services.llm_service import LLMSynthesisService
    from backend.app.services.agent_service import AgentGraphState
except ImportError:
    from services.retriever_service import RetrieverService
    from services.llm_service import LLMSynthesisService
    from services.agent_service import AgentGraphState

# Fallback service instances
retriever_service = RetrieverService()
llm_service = LLMSynthesisService()


def get_retriever_service() -> RetrieverService:
    """Retrieve shared retriever service instance from main app if running."""
    try:
        from backend.app.main import retriever_service as main_retriever
        return main_retriever
    except Exception:
        try:
            from main import retriever_service as main_retriever
            return main_retriever
        except Exception:
            return retriever_service


def get_llm_service() -> LLMSynthesisService:
    """Retrieve shared LLM service instance from main app if running."""
    try:
        from backend.app.main import llm_service as main_llm
        return main_llm
    except Exception:
        try:
            from main import llm_service as main_llm
            return main_llm
        except Exception:
            return llm_service


def classify_route(query: str) -> str:
    """Determines target agent based on query characteristics."""
    visual_keywords = ["image", "chart", "diagram", "figure", "plot", "graph", "picture", "visual", "layout"]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in visual_keywords):
        return "VisionAgent"
    return "SearchAgent"


def supervisor_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Supervisor node that evaluates the query intent and assigns the appropriate target agent.
    """
    question = state.get("question", "")
    target_agent = classify_route(question)

    current_steps = state.get("execution_steps") or []
    intent_desc = "visual/chart analysis" if target_agent == "VisionAgent" else "semantic text retrieval"
    step = {
        "step_number": len(current_steps) + 1,
        "agent_name": "SupervisorAgent",
        "action_taken": "ROUTING_DECISION",
        "details": {
            "route_chosen": target_agent,
            "reasoning": f"Query classified as {intent_desc}.",
            "target_document": state.get("document_id")
        }
    }

    print(f"[Supervisor] routed to: {target_agent}")

    return {
        "current_agent": target_agent,
        "execution_steps": [step]
    }


def search_agent_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Search agent node that executes real vector retrieval from Qdrant and LLM synthesis.
    """
    question = state.get("question", "")
    document_id = state.get("document_id")

    active_retriever = get_retriever_service()
    active_llm = get_llm_service()

    # Real retrieval call
    chunks = active_retriever.retrieve_relevant_chunks(
        query=question,
        top_k=3,
        document_id=document_id
    )

    # Real LLM synthesis call
    answer = active_llm.generate_answer(
        question=question,
        retrieved_chunks=chunks
    )

    current_steps = state.get("execution_steps") or []
    step = {
        "step_number": len(current_steps) + 1,
        "agent_name": "SearchAgent",
        "action_taken": "HYBRID_VECTOR_RETRIEVAL",
        "details": {
            "status": "RETRIEVED_CHUNKS",
            "chunks_count": len(chunks)
        }
    }

    return {
        "current_agent": "SearchAgent",
        "retrieved_chunks": chunks,
        "final_answer": answer,
        "status": "COMPLETED",
        "execution_steps": [step]
    }


def vision_agent_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Vision agent stub explicitly labeled as not implemented.
    """
    current_steps = state.get("execution_steps") or []
    step = {
        "step_number": len(current_steps) + 1,
        "agent_name": "VisionAgent",
        "action_taken": "VLM_IMAGE_REASONING_STUB",
        "details": {
            "status": "NOT_IMPLEMENTED",
            "message": "VisionAgent is an explicit stub in this release."
        }
    }

    return {
        "current_agent": "VisionAgent",
        "retrieved_chunks": [],
        "final_answer": "[Vision Agent Notice] Vision and chart reasoning module is not yet implemented in this release.",
        "status": "NOT_IMPLEMENTED",
        "execution_steps": [step]
    }


def router(state: AgentGraphState) -> Literal["search_agent", "vision_agent"]:
    """Conditional routing edge function based on supervisor decision."""
    current_agent = state.get("current_agent")
    if current_agent == "VisionAgent":
        return "vision_agent"
    return "search_agent"


# Build LangGraph StateGraph
workflow = StateGraph(AgentGraphState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("search_agent", search_agent_node)
workflow.add_node("vision_agent", vision_agent_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "search_agent": "search_agent",
        "vision_agent": "vision_agent"
    }
)

workflow.add_edge("search_agent", END)
workflow.add_edge("vision_agent", END)

workflow_graph = workflow.compile()
