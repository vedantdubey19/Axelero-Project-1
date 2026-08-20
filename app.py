import os
import streamlit as st
import requests

# ==========================================
# CONFIG
# ==========================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ==========================================
# PAGE
# ==========================================

st.set_page_config(
    page_title="OmniBrain",
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


# ==========================================
# HEADER
# ==========================================

st.title("🧠 OmniBrain")
st.write("Upload your PDF and ask questions")


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📄 Document")

    if st.session_state.uploaded_filename:

        st.success("PDF Ready")

        st.write(
            st.session_state.uploaded_filename
        )

    else:

        st.info(
            "Upload a PDF to start."
        )


    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()


# ==========================================
# PDF UPLOAD
# ==========================================

st.subheader("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Upload PDF", type="primary"):

        with st.spinner(
            "📤 Uploading PDF..."
        ):

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

                    st.session_state.uploaded_filename = (
                        uploaded_file.name
                    )

                    st.success(
                        "✅ PDF uploaded successfully!"
                    )

                else:

                    st.error(
                        f"❌ Upload failed: "
                        f"{response.status_code}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Backend is not running."
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


# ==========================================
# CURRENT DOCUMENT
# ==========================================

if st.session_state.uploaded_filename:

    st.caption(
        f"📄 Current document: "
        f"{st.session_state.uploaded_filename}"
    )


# ==========================================
# CHAT
# ==========================================

st.divider()

st.subheader("💬 Chat with your document")


# ==========================================
# DISPLAY OLD CHAT
# ==========================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**{source['filename']}**  \n"
                        f"Chunk: `{source['chunk_id']}`  \n"
                        f"Distance: `{source['distance']}`"
                    )


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask something about your PDF..."
)


# ==========================================
# PROCESS QUESTION
# ==========================================

if question:

    # --------------------------------------
    # User Message
    # --------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------
    # Assistant
    # --------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching your document..."
        ):

            try:

                response = requests.get(
                    f"{BACKEND_URL}/api/v1/search",
                    params={
                        "query": question,
                        "top_k": 3
                    },
                    timeout=60
                )


                if response.status_code == 200:

                    data = response.json()

                    matches = data.get(
                        "matches",
                        []
                    )


                    if matches:

                        # ----------------------------------
                        # Create readable answer
                        # ----------------------------------

                        answer_parts = []

                        sources = []


                        for match in matches:

                            text = match.get(
                                "text",
                                ""
                            )

                            metadata = match.get(
                                "metadata",
                                {}
                            )

                            filename = metadata.get(
                                "filename",
                                "Unknown"
                            )

                            chunk_id = metadata.get(
                                "chunk_id",
                                "Unknown"
                            )

                            distance = match.get(
                                "distance",
                                0
                            )


                            # Save text

                            answer_parts.append(
                                text
                            )


                            # Save source

                            sources.append(
                                {
                                    "filename": filename,
                                    "chunk_id": chunk_id,
                                    "distance": round(
                                        distance,
                                        4
                                    )
                                }
                            )


                        # ----------------------------------
                        # Display answer
                        # ----------------------------------

                        st.markdown(
                            "Here is the relevant information "
                            "from your document:"
                        )


                        for index, text in enumerate(
                            answer_parts,
                            start=1
                        ):

                            st.markdown(
                                f"**{index}.** {text}"
                            )


                        # ----------------------------------
                        # Sources
                        # ----------------------------------

                        with st.expander(
                            "📚 View Sources"
                        ):

                            for source in sources:

                                st.markdown(
                                    f"**📄 "
                                    f"{source['filename']}**  \n"
                                    f"Chunk: "
                                    f"`{source['chunk_id']}`  \n"
                                    f"Distance: "
                                    f"`{source['distance']}`"
                                )


                        # ----------------------------------
                        # Save assistant message
                        # ----------------------------------

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "Here is the relevant "
                                    "information from your "
                                    "document."
                                ),
                                "sources": sources
                            }
                        )


                    else:

                        st.warning(
                            "No relevant information "
                            "was found."
                        )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "Sorry, I couldn't find "
                                    "relevant information in "
                                    "the uploaded document."
                                ),
                                "sources": []
                            }
                        )


                else:

                    st.error(
                        f"Backend error: "
                        f"{response.status_code}"
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "❌ Request timed out."
                )


            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "OmniBrain • PDF Upload • Semantic Search • Chat"
)