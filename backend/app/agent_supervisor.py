# NOTE: This standalone prototype is superseded by backend.app.agents.supervisor.
# It is kept for historical reference and is no longer part of the active API execution path.

import os
import sys
from typing import TypedDict, Annotated, Sequence, Literal
import operator

# Set root directory and app folder in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Import DDGS directly
try:
    from ddgs import DDGS
    ddg_client = DDGS()
except ImportError:
    ddg_client = None


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str
    context: str


def supervisor_node(state: AgentState) -> AgentState:
    """Decides whether to route to Search or Direct Answer based on query intent."""
    last_message = state["messages"][-1].content.lower()
    search_keywords = ["search", "find", "how", "what is", "document", "rag", "bm25", "vector", "lookup"]
    
    if any(keyword in last_message for keyword in search_keywords):
        state["next_node"] = "search_agent"
    else:
        state["next_node"] = "direct_answer"
        
    return state


def search_agent_node(state: AgentState) -> AgentState:
    """Performs external search via DDGS or local retrieval fallback."""
    query = state["messages"][-1].content
    retrieved_context = ""
    
    if ddg_client:
        try:
            results = list(ddg_client.text(query, max_results=2))
            if results:
                retrieved_context = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        except Exception:
            pass

    if not retrieved_context:
        retrieved_context = f"Retrieved context for knowledge domain query: '{query}'"

    state["context"] = retrieved_context
    state["messages"].append(AIMessage(content=f"[Search Agent Output]:\n{retrieved_context}"))
    return state


def direct_answer_node(state: AgentState) -> AgentState:
    """Responds directly to conversational queries."""
    query = state["messages"][-1].content
    response = f"[Supervisor Response]: Hello! How can I assist you with your project today?"
    state["messages"].append(AIMessage(content=response))
    return state


def router(state: AgentState) -> Literal["search_agent", "direct_answer"]:
    return state["next_node"]


# Build State Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("search_agent", search_agent_node)
workflow.add_node("direct_answer", direct_answer_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "search_agent": "search_agent",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("search_agent", END)
workflow.add_edge("direct_answer", END)

graph_app = workflow.compile()


if __name__ == "__main__":
    print("--- Testing Routing: Search Intent ---")
    search_query = "What is hybrid search in RAG pipelines?"
    inputs = {"messages": [HumanMessage(content=search_query)], "context": "", "next_node": ""}
    for output in graph_app.stream(inputs):
        print(output)

    print("\n--- Testing Routing: Direct Chat Intent ---")
    chat_query = "Hello, good morning!"
    inputs_chat = {"messages": [HumanMessage(content=chat_query)], "context": "", "next_node": ""}
    for output in graph_app.stream(inputs_chat):
        print(output)