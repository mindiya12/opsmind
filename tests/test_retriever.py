"""
tests/test_retriever.py — Retriever and ChromaDB tests

These tests are FAST — no LLM calls, just embedding + ChromaDB search.
They verify the knowledge base was ingested correctly and retrieval
returns meaningful results.

Run just this file:
    pytest tests/test_retriever.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.documents import Document
from rag.retriever import get_retriever, retrieve


class TestRetrieverSetup:
    """Verify the retriever can be constructed and connect to ChromaDB."""

    def test_retriever_builds_without_error(self, retriever):
        """Retriever should construct cleanly from ChromaDB."""
        assert retriever is not None

    def test_retriever_has_correct_k(self, retriever):
        """Retriever should be configured with k=3 as per fixture."""
        assert retriever.k == 3

    def test_retriever_has_collection(self, retriever):
        """ChromaDB collection should exist and have documents."""
        count = retriever.collection.count()
        assert count > 0, (
            f"ChromaDB collection is empty (count={count}). "
            "Run 'python -m ingest.loader' first."
        )


class TestRetrieverResults:
    """Verify retrieval returns correct, relevant results."""

    def test_returns_documents(self, retriever):
        """Any query should return at least one document."""
        docs = retriever.invoke("database error")
        assert len(docs) > 0

    def test_returns_document_objects(self, retriever):
        """Results should be LangChain Document objects."""
        docs = retriever.invoke("connection pool exhausted")
        for doc in docs:
            assert isinstance(doc, Document)
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0

    def test_documents_have_metadata(self, retriever):
        """Every chunk should carry source and doc_type metadata."""
        docs = retriever.invoke("nginx timeout")
        for doc in docs:
            assert "source"   in doc.metadata, "Missing 'source' metadata"
            assert "doc_type" in doc.metadata, "Missing 'doc_type' metadata"

    def test_documents_have_similarity_scores(self, retriever):
        """Retriever should attach similarity scores to metadata."""
        docs = retriever.invoke("disk space warning")
        for doc in docs:
            assert "similarity_score" in doc.metadata
            score = doc.metadata["similarity_score"]
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"

    def test_db_query_returns_db_content(self, retriever):
        """A database query should return at least one chunk from a DB source."""
        docs = retriever.invoke("database connection pool exhausted")
        sources = [d.metadata.get("source", "") for d in docs]
        has_db_source = any(
            "database" in s.lower() or "server_logs" in s.lower()
            for s in sources
        )
        assert has_db_source, (
            f"DB query returned no DB-related sources. Got: {sources}"
        )

    def test_runbook_query_returns_runbook(self, retriever):
        """A 'how to fix' query should surface at least one runbook chunk."""
        docs = retriever.invoke("how to fix nginx upstream timeout 502")
        doc_types = [d.metadata.get("doc_type", "") for d in docs]
        assert "runbook" in doc_types, (
            f"Expected at least one runbook chunk. Got doc_types: {doc_types}"
        )

    def test_different_queries_return_different_results(self, retriever):
        """Semantically different queries should return different top results."""
        db_docs   = retriever.invoke("database connection pool")
        disk_docs = retriever.invoke("disk space /var partition full")

        db_top   = db_docs[0].page_content   if db_docs   else ""
        disk_top = disk_docs[0].page_content if disk_docs else ""

        assert db_top != disk_top, (
            "Different queries returned identical top results — "
            "MMR or embedding may not be working correctly."
        )

    def test_similarity_scores_ordered(self, retriever):
        """First result should have the highest similarity score."""
        docs = retriever.invoke("memory out of memory JVM heap")
        if len(docs) >= 2:
            score_first  = docs[0].metadata.get("similarity_score", 0)
            score_second = docs[1].metadata.get("similarity_score", 0)
            assert score_first >= score_second, (
                f"Results not ordered by score: {score_first} < {score_second}"
            )

    def test_convenience_retrieve_function(self):
        """The standalone retrieve() function should work identically."""
        docs = retrieve("server memory pressure", k=2)
        assert len(docs) <= 2
        assert all(isinstance(d, Document) for d in docs)


class TestKnowledgeBaseContent:
    """Verify specific incidents are retrievable — smoke tests for ingestion."""

    def test_incident_db_pool_is_retrievable(self, retriever):
        """The DB pool exhaustion incident should be in the knowledge base."""
        docs = retriever.invoke("connection pool exhausted max_connections")
        content = " ".join(d.page_content for d in docs)
        assert "connection" in content.lower() and (
            "pool" in content.lower() or "database" in content.lower()
        )

    def test_incident_memory_is_retrievable(self, retriever):
        """The JVM memory spike incident should be in the knowledge base."""
        docs = retriever.invoke("JVM heap GC garbage collection memory")
        content = " ".join(d.page_content for d in docs)
        assert "heap" in content.lower() or "memory" in content.lower()

    def test_incident_disk_is_retrievable(self, retriever):
        """The disk space warning should be in the knowledge base."""
        docs = retriever.invoke("disk space /var partition warning")
        content = " ".join(d.page_content for d in docs)
        assert "disk" in content.lower() or "/var" in content.lower()

    def test_runbook_commands_are_retrievable(self, retriever):
        """Runbook bash commands should be in the knowledge base."""
        docs = retriever.invoke("pg_terminate_backend stale connections")
        content = " ".join(d.page_content for d in docs)
        assert "terminate" in content.lower() or "connection" in content.lower()