from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Splits input text into overlapping chunks with optional metadata tracking.
    
    Args:
        text (str): Raw string content extracted from documents.
        chunk_size (int): Max characters per chunk. Default 1000.
        chunk_overlap (int): Overlap characters between chunks. Default 200.
        metadata (dict): Optional document metadata (e.g., source_id, page_num).
        
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries containing content and metadata.
    """
    if not text or not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )
    
    raw_chunks = text_splitter.split_text(text)
    
    processed_chunks = []
    base_metadata = metadata or {}
    
    for idx, chunk in enumerate(raw_chunks):
        chunk_entry = {
            "chunk_id": idx,
            "text": chunk,
            "char_count": len(chunk),
            "metadata": base_metadata.copy()
        }
        processed_chunks.append(chunk_entry)
        
    return processed_chunks


if __name__ == "__main__":
    # Test sample execution
    sample_doc = """
    Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM responses 
    by grounding the model on external sources of knowledge. RAG models combine a retrieval component 
    with a generative model. 
    
    Chunking is the process of breaking down large documents into smaller, manageable pieces of text. 
    Selecting the right chunk size and overlap is crucial for optimizing semantic search and context quality.
    """
    
    result = chunk_text(sample_doc, chunk_size=150, chunk_overlap=30, metadata={"source": "rag_intro.pdf"})
    print(f"Total Chunks Created: {len(result)}\n")
    for c in result:
        print(f"--- Chunk {c['chunk_id']} ({c['char_count']} chars) ---")
        print(c["text"])
        print()
