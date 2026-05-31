"""
ingest/embeddings.py — Reliable local embedding wrapper.

Calls sentence-transformers directly instead of going through LangChain's
HuggingFaceEmbeddings wrapper, which has known issues on Windows.

This class implements LangChain's Embeddings interface (embed_documents +
embed_query), so it drops in anywhere LangChain expects an embedding object —
including Chroma.from_documents() and the retriever.
"""

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from config import EMBEDDING_MODEL


class LocalEmbeddings(Embeddings):
    """
    Thin wrapper around SentenceTransformer that satisfies LangChain's
    Embeddings interface.

    Why not just use HuggingFaceEmbeddings?
    The langchain-huggingface wrapper adds extra layers that can fail silently
    on Windows, returning empty lists instead of raising a proper error.
    This wrapper calls SentenceTransformer.encode() directly — no hidden layers,
    no silent failures.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"  Loading sentence-transformer model: {model_name}")
        print("  (downloads ~90 MB on first run, cached after that)")
        self.model = SentenceTransformer(model_name)
        print("  Embedding model loaded successfully.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text chunks into a list of vectors.
        Called during ingestion — once per chunk.
        """
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,  # makes cosine similarity more accurate
            show_progress_bar=True,
            batch_size=32,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Convert a single query string into a vector.
        Called at search time — once per user question.
        """
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return vector[0].tolist()