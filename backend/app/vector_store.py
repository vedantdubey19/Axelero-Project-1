import os
import chromadb

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_DIR = "data/raw"
CHROMA_DIR = "data/chroma"

os.makedirs(CHROMA_DIR, exist_ok=True)


model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(
    path=CHROMA_DIR
)


collection = client.get_or_create_collection(
    name="omnibrain_documents"
)


def create_embeddings(filename: str):

    file_path = os.path.join(PDF_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF not found: {filename}"
        )

    reader = PdfReader(file_path)

    full_text = ""

    for page in reader.pages:

        text = page.extract_text() or ""

        text = text.replace("\x00", "")

        full_text += text + "\n"


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )


    chunks = splitter.split_text(full_text)


    if not chunks:
        raise ValueError(
            "No text found in PDF."
        )


    embeddings = model.encode(
        chunks
    ).tolist()


    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]


    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "filename": filename,
                "chunk_id": i + 1
            }
            for i in range(len(chunks))
        ]
    )


    return {
        "filename": filename,
        "total_chunks": len(chunks),
        "embedding_dimension": len(embeddings[0]),
        "collection": "omnibrain_documents"
    }


def search_similar(
    query: str,
    top_k: int = 3
):

    if not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )


    query_embedding = model.encode(
        [query]
    ).tolist()


    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )


    documents = results.get(
        "documents",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]


    matches = []


    for i, document in enumerate(documents):

        matches.append(
            {
                "rank": i + 1,
                "text": document,
                "distance": (
                    distances[i]
                    if i < len(distances)
                    else None
                ),
                "metadata": (
                    metadatas[i]
                    if i < len(metadatas)
                    else {}
                )
            }
        )


    return {
        "query": query,
        "top_k": top_k,
        "matches": matches
    }