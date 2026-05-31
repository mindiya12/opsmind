"""
rag/retriever.py — Knowledge Base Query Interface

Queries ChromaDB by meaning and returns relevant chunks.
Uses ChromaDB's native client directly (same reason as loader.py).

Also provides a LangChain-compatible retriever class so this plugs
straight into the RAG chain on Day 2 and the agents on Days 3-4.

Run with:
    python -m rag.retriever
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Any
import chromadb
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from ingest.embeddings import LocalEmbeddings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, TOP_K_RESULTS


# ── LangChain-compatible retriever class ──────────────────────────────────────

class OpsMindRetriever(BaseRetriever):
    """
    A LangChain BaseRetriever that queries ChromaDB directly.

    Why subclass BaseRetriever?
    LangChain chains and agents expect an object with a .invoke(query) method
    that returns List[Document]. BaseRetriever provides that interface.
    We just implement _get_relevant_documents() and LangChain handles the rest.

    This means our custom ChromaDB setup plugs straight into:
    - RAG chains (Day 2)
    - Agent tools (Days 3-4)
    - Any other LangChain component
    without needing to change those components.
    """

    # Pydantic fields — required because BaseRetriever is a Pydantic model
    collection: Any = Field(description="ChromaDB collection object")
    embeddings_model: Any = Field(description="LocalEmbeddings instance")
    k: int = Field(default=TOP_K_RESULTS, description="Number of results to return")

    class Config:
        arbitrary_types_allowed = True  # Allows non-Pydantic types as fields

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """
        Core retrieval logic. Steps:
        1. Embed the query string → 384-dim vector
        2. Ask ChromaDB to find the k most similar stored vectors
        3. Wrap results as LangChain Document objects (with metadata)
        """
        query_vector = self.embeddings_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        if results and results["documents"] and results["documents"][0]:
            for i, text in enumerate(results["documents"][0]):
                metadata = {}
                if results["metadatas"] and results["metadatas"][0]:
                    metadata = results["metadatas"][0][i]
                # Add similarity score to metadata so agents can see it
                if results["distances"] and results["distances"][0]:
                    # ChromaDB cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity score: 1 = identical, 0 = opposite
                    dist = results["distances"][0][i]
                    metadata["similarity_score"] = round(1 - (dist / 2), 3)

                docs.append(Document(page_content=text, metadata=metadata))

        return docs


# ── Factory function ───────────────────────────────────────────────────────────

def get_retriever(k: int = TOP_K_RESULTS) -> OpsMindRetriever:
    """
    Load ChromaDB and return a ready-to-use retriever.
    Call this from anywhere that needs to query the knowledge base.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_collection(name=CHROMA_COLLECTION)
    embeddings_model = LocalEmbeddings()

    return OpsMindRetriever(
        collection=collection,
        embeddings_model=embeddings_model,
        k=k,
    )


# ── Standalone retrieval function ──────────────────────────────────────────────

def retrieve(query: str, k: int = TOP_K_RESULTS) -> List[Document]:
    """Convenience function — create retriever and run query in one call."""
    return get_retriever(k=k).invoke(query)


# ── Test suite ─────────────────────────────────────────────────────────────────

def test_retrieval():
    test_queries = [
        "database connection pool exhausted",
        "nginx upstream timeout 502 error",
        "how to fix JVM out of memory",
        "disk space warning on /var partition",
    ]

    print("=" * 60)
    print("  OpsMind AI — Retrieval Test")
    print("=" * 60)

    try:
        retriever = get_retriever(k=3)
    except Exception as e:
        print(f"\n[ERROR] Could not load ChromaDB: {e}")
        print("Make sure you've run 'python -m ingest.loader' first.")
        return

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("─" * 60)

        docs = retriever.invoke(query)

        if not docs:
            print("  No results returned.")
            continue

        for i, doc in enumerate(docs, 1):
            source   = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("doc_type", "unknown")
            score    = doc.metadata.get("similarity_score", "?")
            preview  = doc.page_content.strip()[:180].replace("\n", " ")
            print(f"\n  [{i}] {doc_type} — {source}  (similarity: {score})")
            print(f"      {preview}...")

    print(f"\n{'=' * 60}")
    print("  If results match their queries, the knowledge base works!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    test_retrieval()