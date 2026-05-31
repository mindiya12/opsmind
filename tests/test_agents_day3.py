"""
tests/test_agents_day3.py — Day 3 integration test

Manually threads Agent 1 → Agent 2 to verify both work in sequence.
This simulates what LangGraph will automate on Day 4.

Run with:
    python -m tests.test_agents_day3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import OpsState
from agents.retrieval_agent import retrieval_agent
from agents.analyzer_agent  import analyzer_agent


def run_pipeline(query: str) -> OpsState:
    """
    Manually thread Agent 1 → Agent 2.
    On Day 4, LangGraph does this automatically via the StateGraph.
    For now, we do it by hand so we can see every state transition.
    """
    # Initial state — only query is set
    state: OpsState = {
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

    print(f"\nInitial state keys with values: query='{query}'")
    print("All other keys empty — agents will populate them.\n")

    # Agent 1 runs — writes retrieved_docs + formatted_context
    a1_output = retrieval_agent(state)
    state.update(a1_output)
    print(f"\nState after Agent 1:")
    print(f"  retrieved_docs:     {len(state['retrieved_docs'])} documents")
    print(f"  formatted_context:  {len(state['formatted_context'])} chars")

    # Agent 2 runs — reads formatted_context, writes anomalies
    a2_output = analyzer_agent(state)
    state.update(a2_output)
    print(f"\nState after Agent 2:")
    print(f"  anomalies:          {len(state['anomalies'])} detected")
    print(f"  has_anomalies:      {state['has_anomalies']}")
    print(f"  anomaly_summary:    {state['anomaly_summary']}")

    return state


if __name__ == "__main__":
    test_queries = [
        "What went wrong with the database overnight?",
        "Are there any memory or performance issues in the logs?",
    ]

    print("=" * 65)
    print("  Day 3 — Agent 1 + Agent 2 Integration Test")
    print("=" * 65)

    for query in test_queries:
        print(f"\n{'─'*65}")
        print(f"QUERY: {query}")
        print(f"{'─'*65}")

        final_state = run_pipeline(query)

        print(f"\n{'─'*65}")
        print(f"FINAL STATE SUMMARY")
        print(f"{'─'*65}")
        print(f"  Anomalies found: {len(final_state['anomalies'])}")

        for a in final_state["anomalies"]:
            sev = a.get("severity", "?").upper()
            typ = a.get("anomaly_type", "?")
            svc = a.get("service", "?")
            ts  = a.get("timestamp", "")
            ts_str = f" @ {ts}" if ts else ""
            print(f"  [{sev}] {typ} in {svc}{ts_str}")

        if final_state.get("error"):
            print(f"  ERROR: {final_state['error']}")

        print(f"\n  → Routing decision: "
              f"{'send to Agent 3 (Solution Generator)' if final_state['has_anomalies'] else 'skip Agent 3 (no anomalies)'}")

    print(f"\n{'=' * 65}")
    print("  Day 3 complete. Both agents working.")
    print("  Next: Day 4 — wire into LangGraph StateGraph with")
    print("  Agent 3 (Solution Generator) and conditional routing.")
    print(f"{'=' * 65}\n")