import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """Represents an intermediate reasoning step taken by a graph node."""
    step_number: int
    agent_name: str
    action_taken: str
    details: Optional[Dict[str, Any]] = None


class AgentQueryRequest(BaseModel):
    """Request schema for agentic queries."""
    question: str = Field(..., min_length=2, description="The user query to be routed by the supervisor.")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Stateful session ID.")
    document_id: Optional[str] = Field(default=None, description="Optional document filter.")


class AgentQueryResponse(BaseModel):
    """Standardized multi-step response payload for frontend visualization."""
    query_id: str
    session_id: str
    question: str
    final_answer: str
    execution_steps: List[AgentStep]
    referenced_sources: List[Dict[str, Any]]
    status: str


class AgentOrchestrationService:
    """
    Service layer bridging FastAPI to LangGraph compiled graphs.
    """

    def __init__(self):
        self.active_sessions: Dict[str, List[Dict[str, Any]]] = {}

    async def execute_agent_workflow(self, question: str, session_id: str, document_id: Optional[str] = None) -> Dict[
        str, Any]:
        """
        Executes or simulates the multi-agent graph run and returns the trace.
        """
        steps: List[AgentStep] = []

        # Step 1: Supervisor Routing Decision
        steps.append(AgentStep(
            step_number=1,
            agent_name="SupervisorAgent",
            action_taken="ROUTING_DECISION",
            details={"route_chosen": "SearchAgent", "reasoning": "Query requires text/table document retrieval."}
        ))

        # Step 2: Search Agent Tool Invocation
        steps.append(AgentStep(
            step_number=2,
            agent_name="SearchAgent",
            action_taken="VECTOR_SEARCH_RETRIEVAL",
            details={"chunks_found": 2, "filter_applied": document_id}
        ))

        # Step 3: Synthesis & Citation Verification
        steps.append(AgentStep(
            step_number=3,
            agent_name="SupervisorAgent",
            action_taken="SYNTHESIS_AND_FINALIZE",
            details={"status": "ANSWER_READY"}
        ))

        mock_answer = (
            f"Supervisor routed your query through the Search Agent. "
            f"Here is the synthesized response to: '{question}'."
        )

        sources = [
            {"source": document_id or "data/raw/uploaded_doc.pdf", "page": 1, "score": 0.91}
        ]

        return {
            "query_id": str(uuid.uuid4()),
            "session_id": session_id,
            "question": question,
            "final_answer": mock_answer,
            "execution_steps": steps,
            "referenced_sources": sources,
            "status": "COMPLETED"
        }