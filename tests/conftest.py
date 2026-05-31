"""
tests/conftest.py — Shared pytest fixtures

Fixtures are reusable setup functions that pytest injects into tests
that declare them as parameters. Instead of every test building its
own initial state or retriever, they declare the fixture name and
pytest handles the rest.

Scope explanation:
    scope="session" — fixture runs ONCE for the entire test session.
                      Used for expensive objects like the pipeline and
                      retriever that are slow to build.
    scope="function" — fixture runs fresh for every test (default).
                       Used for state objects that tests might mutate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from graph.state import OpsState


# ── Shared initial state ───────────────────────────────────────────────────────

def make_initial_state(query: str) -> OpsState:
    """Build a fresh OpsState with only the query set."""
    return {
        "query":             query,
        "retrieved_docs":    [],
        "formatted_context": "",
        "anomalies":         [],
        "has_anomalies":     False,
        "anomaly_summary":   "",
        "recommendations":   None,
        "final_answer":      None,
        "error":             None,
    }


@pytest.fixture(scope="session")
def retriever():
    """Build the retriever once for the whole test session."""
    from rag.retriever import get_retriever
    return get_retriever(k=3)


@pytest.fixture(scope="session")
def pipeline():
    """Compile the LangGraph pipeline once for the whole test session."""
    from graph.pipeline import build_pipeline
    return build_pipeline()


@pytest.fixture
def db_query_state() -> OpsState:
    return make_initial_state("What happened with the database connection pool?")


@pytest.fixture
def memory_query_state() -> OpsState:
    return make_initial_state("Are there any JVM memory or GC pressure issues?")


@pytest.fixture
def disk_query_state() -> OpsState:
    return make_initial_state("Is there a disk space problem on /var?")