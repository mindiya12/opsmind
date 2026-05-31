"""
agents/retrieval_agent.py — Agent 1: The Retrieval Agent

Job: Take the user's question, search ChromaDB, return relevant context.
Nothing more. No LLM calls. No analysis. Just retrieval.

Why keep retrieval as a separate agent instead of folding it into Agent 2?
Two reasons:
    1. Separation of concerns — if retrieval fails, the error is isolated here,
       not buried inside a more complex agent.
    2. In Day 4, LangGraph can route AROUND this agent if context was already
       retrieved (e.g. in a follow-up question in the same session). Having it
       separate makes that optimization easy to add.

LangGraph node contract:
    - Input:  OpsState (reads: query)
    - Output: partial dict (writes: retrieved_docs, formatted_context)
    - LangGraph merges the returned dict back into the full state.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import OpsState
from rag.retriever import get_retriever
from rag.chain import format_context
from config import TOP_K_RESULTS


def retrieval_agent(state: OpsState) -> dict:
    """
    Agent 1 node function.

    Reads from state:  query
    Writes to state:   retrieved_docs, formatted_context

    Returns only the keys it updates — LangGraph merges these
    into the full state automatically.
    """
    query = state["query"]
    print(f"\n[Agent 1 — Retrieval] Searching for: '{query[:60]}...' " if len(query) > 60 
          else f"\n[Agent 1 — Retrieval] Searching for: '{query}'")

    try:
        retriever = get_retriever(k=TOP_K_RESULTS)
        docs = retriever.invoke(query)
        context = format_context(docs)

        print(f"[Agent 1 — Retrieval] Found {len(docs)} relevant chunks:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            score  = doc.metadata.get("similarity_score", "?")
            print(f"    [{i}] {source} (similarity: {score})")

        return {
            "retrieved_docs":    docs,
            "formatted_context": context,
        }

    except Exception as e:
        print(f"[Agent 1 — Retrieval] ERROR: {e}")
        return {
            "retrieved_docs":    [],
            "formatted_context": "No context retrieved due to an error.",
            "error":             f"Retrieval failed: {str(e)}",
        }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_state: OpsState = {
        "query":             "Why did the database fail at 3am?",
        "retrieved_docs":    [],
        "formatted_context": "",
        "anomalies":         [],
        "has_anomalies":     False,
        "anomaly_summary":   "",
        "recommendations":   None,
        "final_answer":      None,
        "error":             None,
    }

    print("=" * 55)
    print("  Agent 1 — Retrieval Agent standalone test")
    print("=" * 55)

    result = retrieval_agent(test_state)

    print(f"\nReturned keys: {list(result.keys())}")
    print(f"Docs retrieved: {len(result['retrieved_docs'])}")
    print("\nFormatted context preview (first 500 chars):")
    print(result["formatted_context"][:500])