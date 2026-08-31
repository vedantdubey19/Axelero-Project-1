import uuid
from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator
from pydantic import BaseModel, Field

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


# --- Shared LangGraph State Schema ---

class AgentGraphState(TypedDict):
    """Shared state contract between LangGraph nodes."""
    question: str
    session_id: str
    document_id: Optional[str]
    current_agent: str
    retrieved_chunks: List[Dict[str, Any]]
    execution_steps: Annotated[List[Dict[str, Any]], operator.add]
    final_answer: str
    status: str


# --- Pydantic Schemas for API Layer ---

class AgentStep(BaseModel):
    step_number: int
    agent_name: str
    action_taken: str
    details: Optional[Dict[str, Any]] = None


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="The user query to be routed by the supervisor.")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Stateful session ID.")
    document_id: Optional[str] = Field(default=None, description="Optional target document filter.")


class AgentQueryResponse(BaseModel):
    query_id: str
    session_id: str
    question: str
    routed_agent: str
    final_answer: str
    execution_steps: List[AgentStep]
    referenced_sources: List[Dict[str, Any]]
    status: str


# --- Agent Service Layer ---

class AgentOrchestrationService:
    """
    Day 14 Implementation: Multi-agent supervisor router supporting dynamic
    dispatch between SearchAgent and VisionAgent.
    """
    def __init__(self):
        self.compiled_graph = None
        self._initialize_graph()

    def _initialize_graph(self):
        try:
            from backend.app.agents.supervisor import workflow_graph
            self.compiled_graph = workflow_graph
        except ImportError:
            try:
                from agents.supervisor import workflow_graph
                self.compiled_graph = workflow_graph
            except ImportError:
                self.compiled_graph = None

    def _classify_route(self, query: str) -> str:
        """Determines target agent based on query characteristics."""
        visual_keywords = ["image", "chart", "diagram", "figure", "plot", "graph", "picture", "visual", "layout"]
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in visual_keywords):
            return "VisionAgent"
        return "SearchAgent"

    @tracing_service.observe(name="agent_workflow_execution")
    async def execute_agent_workflow(
        self,
        question: str,
        session_id: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-agent graph run with dynamic routing and step tracing.
        """
        if not self.compiled_graph:
            self._initialize_graph()

        if not self.compiled_graph:
            raise RuntimeError(
                "Agent supervisor workflow graph is not initialized. "
                "Ensure backend.app.agents.supervisor.workflow_graph is accessible."
            )

        initial_state: AgentGraphState = {
            "question": question,
            "session_id": session_id,
            "document_id": document_id,
            "current_agent": "SupervisorAgent",
            "retrieved_chunks": [],
            "execution_steps": [],
            "final_answer": "",
            "status": "RUNNING"
        }

        try:
            final_state = await self.compiled_graph.ainvoke(initial_state)
        except Exception as e:
            raise RuntimeError(f"Agent workflow execution failed: {str(e)}")

        routed_to = final_state.get("current_agent", "SearchAgent")
        raw_steps = final_state.get("execution_steps", [])
        formatted_steps: List[AgentStep] = []
        for idx, step in enumerate(raw_steps):
            if isinstance(step, AgentStep):
                formatted_steps.append(step)
            elif isinstance(step, dict):
                formatted_steps.append(
                    AgentStep(
                        step_number=step.get("step_number", idx + 1),
                        agent_name=step.get("agent_name", "UnknownAgent"),
                        action_taken=step.get("action_taken", "UNKNOWN"),
                        details=step.get("details")
                    )
                )

        return {
            "query_id": str(uuid.uuid4()),
            "session_id": session_id,
            "question": question,
            "routed_agent": routed_to,
            "final_answer": final_state.get("final_answer", ""),
            "execution_steps": formatted_steps,
            "referenced_sources": final_state.get("retrieved_chunks", []),
            "status": final_state.get("status", "COMPLETED")
        }