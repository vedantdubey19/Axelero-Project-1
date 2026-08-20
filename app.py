import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OmniBrain",
    page_icon="🧠",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "job_id" not in st.session_state:
    st.session_state.job_id = None

st.title("🧠 OmniBrain")
st.write("Upload your PDF and ask questions")

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


st.subheader("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Upload PDF", type="primary"):

        with st.spinner("📤 Uploading PDF..."):

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

                    data = response.json()

                    job_id = data.get("job_id")

                    if not job_id:

                        st.error(
                            "❌ Upload succeeded, but no job ID was returned."
                        )

                    else:

                        st.session_state.uploaded_filename = (
                            uploaded_file.name
                        )

                        st.session_state.job_id = job_id

                        st.success(
                            "✅ PDF uploaded successfully!"
                        )

                        status_placeholder = st.empty()

                        while True:

                            try:

                                status_response = requests.get(
                                    f"{BACKEND_URL}/api/v1/ingest/status/{job_id}",
                                    timeout=30
                                )

                                if status_response.status_code != 200:

                                    status_placeholder.error(
                                        f"❌ Unable to check processing status: "
                                        f"{status_response.status_code}"
                                    )

                                    break

                                status_data = status_response.json()

                                status = status_data.get(
                                    "status",
                                    "UNKNOWN"
                                )

                                if status == "QUEUED":

                                    status_placeholder.info(
                                        "⏳ PDF is queued for processing..."
                                    )

                                elif status == "PROCESSING":

                                    status_placeholder.info(
                                        "⚙️ PDF is being processed..."
                                    )

                                elif status in ["COMPLETED", "DONE"]:

                                    status_placeholder.success(
                                        "✅ PDF processing completed. "
                                        "Your document is ready!"
                                    )

                                    break

                                elif status == "FAILED":

                                    status_placeholder.error(
                                        "❌ PDF processing failed."
                                    )

                                    break

                                else:

                                    status_placeholder.warning(
                                        f"ℹ️ Current processing status: "
                                        f"{status}"
                                    )

                                time.sleep(2)

                            except requests.exceptions.Timeout:

                                status_placeholder.warning(
                                    "⏳ Waiting for processing status..."
                                )

                            except requests.exceptions.ConnectionError:

                                status_placeholder.error(
                                    "❌ Cannot connect to FastAPI."
                                )

                                break

                            except Exception as e:

                                status_placeholder.error(
                                    f"❌ Status check error: {e}"
                                )

                                break

                else:

                    st.error(
                        f"❌ Upload failed: "
                        f"{response.status_code}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Backend is not running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "❌ Upload request timed out."
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )


if st.session_state.uploaded_filename:

    st.caption(
        f"📄 Current document: "
        f"{st.session_state.uploaded_filename}"
    )


st.divider()

st.subheader("💬 Chat with your document")


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
                        f"**📄 {source['filename']}**  \n"
                        f"Chunk: `{source['chunk_id']}`  \n"
                        f"Page: `{source['page']}`  \n"
                        f"Score: `{source['score']}`"
                    )


question = st.chat_input(
    "Ask something about your PDF..."
)


if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching your document..."
        ):

            try:

                response = requests.post(
                    f"{BACKEND_URL}/api/v1/query",
                    json={
                        "question": question,
                        "top_k": 3,
                        "document_id": None
                    },
                    timeout=60
                )


                if response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        ""
                    )

                    retrieved_chunks = data.get(
                        "retrieved_chunks",
                        []
                    )

                    status = data.get(
                        "status",
                        "UNKNOWN"
                    )


                    if status == "SUCCESS":

                        if answer:

                            st.markdown(answer)

                        else:

                            st.warning(
                                "⚠️ The backend returned an empty answer."
                            )


                        sources = []

                        for chunk in retrieved_chunks:

                            chunk_id = chunk.get(
                                "chunk_id",
                                "Unknown"
                            )

                            content = chunk.get(
                                "content",
                                ""
                            )

                            page = chunk.get(
                                "page",
                                "Unknown"
                            )

                            score = chunk.get(
                                "score",
                                0
                            )

                            source = chunk.get(
                                "source",
                                "Unknown"
                            )

                            sources.append(
                                {
                                    "filename": source,
                                    "chunk_id": chunk_id,
                                    "page": page,
                                    "score": round(
                                        score,
                                        4
                                    ),
                                    "content": content
                                }
                            )


                        if sources:

                            with st.expander(
                                "📚 View Sources"
                            ):

                                for source in sources:

                                    st.markdown(
                                        f"**📄 "
                                        f"{source['filename']}**  \n"
                                        f"Chunk: "
                                        f"`{source['chunk_id']}`  \n"
                                        f"Page: "
                                        f"`{source['page']}`  \n"
                                        f"Score: "
                                        f"`{source['score']}`"
                                    )

                                    if source["content"]:

                                        st.caption(
                                            source["content"]
                                        )

                                    st.divider()


                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer
                                if answer
                                else "No answer was returned.",
                                "sources": sources
                            }
                        )


                    else:

                        st.warning(
                            f"⚠️ Query completed with status: "
                            f"{status}"
                        )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": (
                                    f"Query status: {status}"
                                ),
                                "sources": []
                            }
                        )


                elif response.status_code == 422:

                    st.error(
                        "❌ Invalid request sent to the backend."
                    )


                else:

                    st.error(
                        f"❌ Backend error: "
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


st.divider()

st.caption(
    "OmniBrain • PDF Upload • Semantic Search • Chat"
)