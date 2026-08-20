import os
import uuid
import streamlit as st
import requests

# ==========================================
# CONFIG
# ==========================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="OmniBrain • Multi-Agent RAG",
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ==========================================
# HEADER
# ==========================================

st.title("🧠 OmniBrain")
st.caption("Multi-Agent Document Intelligence & LangGraph Supervisor Orchestration")


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("📄 Document Management")

    if st.session_state.uploaded_filename:
        st.success(f"✅ Active: **{st.session_state.uploaded_filename}**")
    else:
        st.info("Upload a PDF document to begin asking questions.")

    st.divider()

    st.caption(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.caption(f"**Backend Gateway:** `{BACKEND_URL}`")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ==========================================
# PDF UPLOAD WIDGET
# ==========================================

st.subheader("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Upload & Index Document", type="primary"):
        with st.spinner("📤 Uploading and triggering background ingestion pipeline..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/api/v1/upload",
                    files=files,
                    timeout=120
                )

                if response.status_code in [200, 201]:
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success(f"✅ Document **{uploaded_file.name}** uploaded and queued for vector indexing!")
                else:
                    st.error(f"❌ Upload failed with status code {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI backend at " + BACKEND_URL)
            except Exception as e:
                st.error(f"❌ Error during upload: {e}")


# ==========================================
# CURRENT DOCUMENT STATUS BANNER
# ==========================================

if st.session_state.uploaded_filename:
    st.caption(f"📌 Context Filter: **{st.session_state.uploaded_filename}**")

st.divider()


# ==========================================
# CHAT INTERFACE
# ==========================================

st.subheader("💬 Chat with OmniBrain Multi-Agent Assistant")


# ------------------------------------------
# Render Past Messages
# ------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display past agent steps
            if message.get("execution_steps"):
                routed = message.get("routed_agent", "Agent")
                with st.expander(f"🤖 Handled by {routed} (View Decision Trace)", expanded=False):
                    for step in message["execution_steps"]:
                        agent_name = step.get("agent_name", "")
                        details = step.get("details") or {}
                        if agent_name == "SupervisorAgent":
                            route = details.get("route_chosen", routed)
                            reason = details.get("reasoning", "")
                            st.markdown(f"- 🧭 **SupervisorAgent**: Routed to `{route}` *({reason})*")
                        elif agent_name == "SearchAgent":
                            count = details.get("chunks_count", len(message.get("sources", [])))
                            st.markdown(f"- 🔍 **SearchAgent**: Retrieved `{count}` chunk(s) via hybrid vector search")
                        elif agent_name == "VisionAgent":
                            msg = details.get("message", "Visual reasoning stub.")
                            st.markdown(f"- 👁️ **VisionAgent**: `{msg}`")
                        else:
                            st.markdown(f"- ⚙️ **{agent_name}**: `{step.get('action_taken')}`")

            # Vision stub notice if applicable
            if message.get("routed_agent") == "VisionAgent" and message.get("status") == "NOT_IMPLEMENTED":
                st.info("ℹ️ **Vision Agent Notice**: Multimodal chart/image reasoning is an explicit labeled stub in this release.")

            # Main text answer
            st.markdown(message["content"])

            # Sources accordion
            if message.get("sources"):
                with st.expander(f"📚 Sources & Citations ({len(message['sources'])} passages)", expanded=False):
                    for idx, source in enumerate(message["sources"], start=1):
                        fn = source.get("source") or source.get("filename", "Unknown Document")
                        page_num = source.get("page", 1)
                        score = source.get("score")
                        score_str = f" • Score: `{score:.4f}`" if score is not None else ""
                        content = source.get("content") or source.get("text", "")
                        st.markdown(f"**{idx}. 📄 {fn}** (Page {page_num}{score_str})")
                        st.caption(f"\"{content}\"")
        else:
            st.markdown(message["content"])


# ------------------------------------------
# Chat Input & Real-Time Agent Execution
# ------------------------------------------

question = st.chat_input("Ask a question about your uploaded document or request chart analysis...")

if question:
    # 1. Record and display user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # 2. Assistant execution via LangGraph Supervisor
    with st.chat_message("assistant"):
        with st.spinner("🧠 Supervisor evaluating query intent and orchestrating specialized agents..."):
            try:
                payload = {
                    "question": question,
                    "session_id": st.session_state.session_id,
                    "document_id": st.session_state.uploaded_filename
                }

                response = requests.post(
                    f"{BACKEND_URL}/api/v1/agent/query",
                    json=payload,
                    timeout=90
                )

                if response.status_code == 200:
                    data = response.json()

                    routed_agent = data.get("routed_agent", "SearchAgent")
                    final_answer = data.get("final_answer", "")
                    execution_steps = data.get("execution_steps", [])
                    referenced_sources = data.get("referenced_sources", [])
                    agent_status = data.get("status", "COMPLETED")

                    # Live visual execution steps container
                    with st.status(f"🤖 Handled by {routed_agent}", expanded=True) as status_box:
                        for step in execution_steps:
                            agent_name = step.get("agent_name", "")
                            action = step.get("action_taken", "")
                            details = step.get("details") or {}

                            if agent_name == "SupervisorAgent":
                                route = details.get("route_chosen", routed_agent)
                                reason = details.get("reasoning", "")
                                st.write(f"🧭 **Supervisor Decision**: Routed to `{route}`")
                                st.caption(f"Reasoning: {reason}")
                            elif agent_name == "SearchAgent":
                                count = details.get("chunks_count", len(referenced_sources))
                                st.write(f"🔍 **Search Agent**: Executed vector retrieval ({count} passages found)")
                            elif agent_name == "VisionAgent":
                                msg = details.get("message", "Visual reasoning stub.")
                                st.write(f"👁️ **Vision Agent**: {msg}")
                            else:
                                st.write(f"⚙️ **{agent_name}**: {action}")

                        status_box.update(label=f"🤖 Handled by {routed_agent}", state="complete", expanded=False)

                    # Explicit vision stub notice if applicable
                    if routed_agent == "VisionAgent" and agent_status == "NOT_IMPLEMENTED":
                        st.info("ℹ️ **Vision Agent Notice**: Multimodal chart/image reasoning is an explicit labeled stub in this release.")

                    # Main Final Answer
                    st.markdown(final_answer)

                    # Grounded Sources Accordion
                    if referenced_sources:
                        with st.expander(f"📚 Sources & Citations ({len(referenced_sources)} passages)", expanded=False):
                            for idx, source in enumerate(referenced_sources, start=1):
                                fn = source.get("source", "Document")
                                page_num = source.get("page", 1)
                                score = source.get("score", 0.0)
                                content = source.get("content", "")
                                st.markdown(f"**{idx}. 📄 {fn}** (Page {page_num} • Score: `{score:.4f}`)")
                                st.caption(f"\"{content}\"")

                    # Save to session history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_answer,
                        "routed_agent": routed_agent,
                        "status": agent_status,
                        "execution_steps": execution_steps,
                        "sources": referenced_sources
                    })

                else:
                    st.error(f"❌ Backend error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI backend at " + BACKEND_URL)
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out waiting for agent workflow.")
            except Exception as e:
                st.error(f"❌ Error during query processing: {e}")


# ==========================================
# FOOTER
# ==========================================

st.divider()
st.caption("OmniBrain • Multi-Agent RAG • LangGraph Supervisor • Qdrant Vector DB")