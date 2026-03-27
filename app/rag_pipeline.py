"""
RAG Pipeline — PDF ingestion, chunking, embedding, and retrieval.
Uses:
  - pypdf for text extraction
  - sentence-transformers (all-MiniLM-L6-v2) for local embeddings
  - FAISS as the vector store
"""

import os
import io
import re
from typing import List, Tuple

import streamlit as st

# ── Lazy imports so the app still loads if these are missing ──────────────────
try:
    from pypdf import PdfReader
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    _EMBED_OK = True
except ImportError:
    _EMBED_OK = False

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove excess whitespace from extracted text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    if not _PYPDF_OK:
        return "pypdf is not installed — cannot extract PDF text."
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(_clean(t))
    return "\n\n".join(p for p in pages if p)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    Tries to split on sentence boundaries first; falls back to character split.
    """
    # Split into sentences (rough)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            # Start new chunk with overlap
            words = current.split()
            overlap_words = words[-overlap // 5:] if overlap and words else []
            current = " ".join(overlap_words) + (" " if overlap_words else "") + sentence

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 30]  # Filter trivially short chunks


# ── In-memory vector store using FAISS ───────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_embedding_model():
    """Load and cache the sentence-transformer model."""
    if not _EMBED_OK:
        return None
    return SentenceTransformer(EMBEDDING_MODEL)


class VectorStore:
    """Lightweight FAISS-backed vector store for RAG."""

    def __init__(self):
        self.chunks: List[str] = []
        self.index = None
        self.model = _load_embedding_model()

    def _embed(self, texts: List[str]):
        if self.model is None:
            return None
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def add_documents(self, chunks: List[str]):
        """Embed chunks and add to the FAISS index."""
        if not _EMBED_OK or not chunks:
            return False

        embeddings = self._embed(chunks)
        if embeddings is None:
            return False

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)
        return True

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """Return the top-k most relevant chunks for a query."""
        if not _EMBED_OK or self.index is None or not self.chunks:
            return []

        q_emb = self._embed([query]).astype("float32")
        distances, indices = self.index.search(q_emb, min(k, len(self.chunks)))
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def chunk_count(self) -> int:
        return len(self.chunks)

    def clear(self):
        self.chunks = []
        self.index = None


def build_rag_context(vector_store: VectorStore, query: str, max_chars: int = 2000) -> str:
    """Retrieve relevant chunks and assemble a context string for the LLM."""
    chunks = vector_store.similarity_search(query, k=4)
    if not chunks:
        return ""
    context = "\n\n---\n\n".join(chunks)
    return context[:max_chars]
