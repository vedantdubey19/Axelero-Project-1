import streamlit as st
import fitz

st.title("OmniBrain")

st.write("Upload your PDF and ask questions")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("PDF uploaded successfully!")

    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    st.subheader("PDF Content")
    st.write(text)