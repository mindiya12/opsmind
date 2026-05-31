"""
graph/state.py — The Shared State Schema

This TypedDict is the single most important architectural decision in the
multi-agent system. Every agent reads from and writes to this object.
It is the only way agents communicate — no agent calls another directly.

Why TypedDict instead of a regular dict?
TypedDict gives you type hints and IDE autocomplete while remaining
a plain Python dict at runtime. LangGraph uses it to validate that
agents only write keys that exist in the schema.

Think of OpsState as a baton in a relay race — each agent picks it up,
does their job, adds their output, and passes it on. LangGraph manages
the passing; you just define what the baton contains.
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class AnomalyFlag(TypedDict):
    """
    A single detected anomaly — the structured output of Agent 2.
    Using TypedDict for this too means Agent 3 gets type-safe access
    to each field: anomaly["severity"], anomaly["evidence"], etc.
    """
    id:            str  # "anomaly_1", "anomaly_2" — for referencing in Agent 3
    anomaly_type:  str  # "connection_pool_exhaustion" | "memory_pressure" | etc.
    severity:      str  # "critical" | "high" | "medium" | "low"
    service:       str  # "database" | "nginx" | "jvm" | "system" | etc.
    timestamp:     str  # from the log line, e.g. "03:42:20" — empty if not found
    description:   str  # one sentence: what happened
    evidence:      str  # the specific log line that is the strongest proof
    impact:        str  # what downstream services or users were affected


class OpsState(TypedDict):
    """
    The complete shared state passed through the LangGraph pipeline.

    Populated progressively as agents run:
        User query           → query is set
        Agent 1 runs         → retrieved_docs + formatted_context set
        Agent 2 runs         → anomalies + has_anomalies + anomaly_summary set
        Agent 3 runs (Day 4) → recommendations + final_answer set

    Optional fields use Optional[X] to signal they may be empty
    at certain points in the pipeline (before the relevant agent runs).
    """
    # ── Input ────────────────────────────────────────────────────────────────
    query:              str                    # The user's question — set at entry

    # ── Agent 1 output ────────────────────────────────────────────────────────
    retrieved_docs:     List[Document]         # Raw LangChain Document objects
    formatted_context:  str                    # Labelled string ready for LLM prompt

    # ── Agent 2 output ────────────────────────────────────────────────────────
    anomalies:          List[AnomalyFlag]      # Structured anomaly objects
    has_anomalies:      bool                   # Routing flag: True → go to Agent 3
    anomaly_summary:    str                    # One-sentence summary of all anomalies

    # ── Agent 3 output (Day 4) ────────────────────────────────────────────────
    recommendations:    Optional[List[dict]]   # Structured fix recommendations
    final_answer:       Optional[str]          # The complete response to the user

    # ── Error tracking ────────────────────────────────────────────────────────
    error:              Optional[str]          # Set if any agent fails gracefully