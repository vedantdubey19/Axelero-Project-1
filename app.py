import streamlit as st

st.title("OmniBrain")

st.write("Upload your PDF and ask questions")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success("PDF uploaded successfully!")
