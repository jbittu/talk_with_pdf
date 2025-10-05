from typing import List, Tuple
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_ID

# Lazy-load embedder (single instance)
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
    return _embedder

def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 120, respect_pages: bool = True) -> List[str]:
    """Chunk text into smaller pieces for embedding.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum number of words per chunk
        overlap: Number of words to overlap between chunks
        respect_pages: Whether to respect page boundaries (assumes [Page X] markers)
        
    Returns:
        List of text chunks
    """
    text = text.replace("\r", " ")
    
    if respect_pages:
        # Split by page markers [Page X]
        page_pattern = r'\[Page \d+\]'
        page_splits = re.split(f'({page_pattern})', text)
        
        # Recombine page markers with their content
        pages = []
        for i in range(0, len(page_splits)-1, 2):
            if i+1 < len(page_splits):
                pages.append(page_splits[i] + page_splits[i+1])
        
        # If there's an odd number of splits, add the last one
        if len(page_splits) % 2 == 1:
            pages.append(page_splits[-1])
        
        # Chunk each page separately
        chunks = []
        for page in pages:
            page_chunks = _chunk_by_words(page, chunk_size, overlap)
            chunks.extend(page_chunks)
        return chunks
    else:
        # Simple word-based chunking
        return _chunk_by_words(text, chunk_size, overlap)

def _chunk_by_words(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Chunk text by word count with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def build_index(text: str) -> Tuple[faiss.IndexFlatIP, List[str], np.ndarray]:
    chunks = _chunk_text(text)
    emb = _get_embedder()
    vectors = emb.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors = vectors / norms
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index, chunks, vectors

def retrieve_context(index_pack, query: str, k: int = 4) -> str:
    index, chunks, _ = index_pack
    emb = _get_embedder()
    q_vec = emb.encode([query], convert_to_numpy=True)
    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(q_vec.astype(np.float32), k)
    retrieved = [chunks[i] for i in I[0] if i >= 0 and i < len(chunks)]
    return "\n\n".join(retrieved)
