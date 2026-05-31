"""
graph/pipeline.py — The LangGraph StateGraph

This is the orchestration layer that wires all three agents into a
directed graph with conditional routing.

Graph structure:

    START
      │
      ▼
  [retrieval_agent]     ← Agent 1: fetch context from ChromaDB
      │
      ▼
  [analyzer_agent]      ← Agent 2: detect anomalies, return structured JSON
      │
      ▼ (conditional edge — reads has_anomalies from state)
      ├── True  → [solution_agent]    → END
      └── False → [direct_answer]     → END

Key LangGraph concepts used here:

1. StateGraph(OpsState)
   The graph is typed to OpsState. Every node receives the full state
   and returns a partial dict that LangGraph merges back in.

2. add_node(name, function)
   Registers a Python function as a graph node. The function signature
   must be: fn(state: OpsState) -> dict

3. add_edge(from, to)
   Unconditional edge — always goes from → to.

4. add_conditional_edges(from, routing_fn, mapping)
   After 'from' runs, calls routing_fn(state) which returns a string key.
   The mapping dict translates that key to the next node name.

5. compile()
   Validates the graph (no orphan nodes, all edges reachable) and returns
   a CompiledGraph object with a .invoke() method.

Once compiled, calling graph.invoke({"query": "..."}) runs the entire
pipeline — all agents, routing, and state merging — in one call.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, START, END

from graph.state import OpsState
from agents.retrieval_agent import retrieval_agent
from agents.analyzer_agent  import analyzer_agent
from agents.solution_agent  import solution_agent, direct_answer_agent


# ── Routing function ───────────────────────────────────────────────────────────

def route_after_analysis(state: OpsState) -> str:
    """
    Conditional routing function — called by LangGraph after analyzer_agent runs.

    Reads has_anomalies from state and returns a string that LangGraph
    uses to look up the next node in the mapping dict below.

    Why return a string instead of the function directly?
    LangGraph's add_conditional_edges() uses a mapping dict so you can
    have more than two branches. Here we have two, but the same pattern
    handles N branches: return "branch_name" → look up in mapping.
    """
    if state.get("has_anomalies", False):
        print("\n[Router] Anomalies detected → routing to Solution Agent")
        return "has_anomalies"
    else:
        print("\n[Router] No anomalies → routing to Direct Answer")
        return "no_anomalies"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_pipeline() -> "CompiledGraph":
    """
    Assemble, validate, and compile the full agent pipeline.

    Returns a CompiledGraph — call .invoke({"query": "..."}) to run it.
    The compiled graph is stateless and thread-safe: you can call .invoke()
    multiple times concurrently without issues.
    """

    # 1. Initialise the graph with our state schema
    graph = StateGraph(OpsState)

    # 2. Register all agent functions as nodes
    #    The string names ("retrieval_agent" etc.) are used in edge definitions
    #    and appear in LangGraph's execution trace for debugging.
    graph.add_node("retrieval_agent",    retrieval_agent)
    graph.add_node("analyzer_agent",     analyzer_agent)
    graph.add_node("solution_agent",     solution_agent)
    graph.add_node("direct_answer",      direct_answer_agent)

    # 3. Unconditional edges — always execute in this order
    graph.add_edge(START,              "retrieval_agent")
    graph.add_edge("retrieval_agent",  "analyzer_agent")

    # 4. Conditional edge after analyzer — branches based on has_anomalies
    graph.add_conditional_edges(
        "analyzer_agent",       # From this node...
        route_after_analysis,   # ...call this function to get a routing key...
        {                       # ...then map the key to the next node:
            "has_anomalies": "solution_agent",
            "no_anomalies":  "direct_answer",
        }
    )

    # 5. Both terminal nodes go to END
    graph.add_edge("solution_agent", END)
    graph.add_edge("direct_answer",  END)

    # 6. Compile — validates graph structure and returns a runnable object
    compiled = graph.compile()
    print("[Pipeline] Graph compiled successfully.")
    return compiled


# ── Convenience wrapper ────────────────────────────────────────────────────────

def run_pipeline(query: str) -> OpsState:
    """
    Build and run the full pipeline for a single query.
    Returns the complete final state with all agent outputs populated.
    """
    pipeline = build_pipeline()

    # Only query needs to be set — agents populate everything else
    initial_state = {
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

    final_state = pipeline.invoke(initial_state)
    return final_state


# ── Full pipeline test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "query":          "What happened with the database at 3:42 AM and how do I fix it?",
            "expect_agents":  "Agent 1 → Agent 2 → Agent 3 (anomalies found)",
        },
        {
            "query":          "Are there any disk space issues I should be aware of?",
            "expect_agents":  "Agent 1 → Agent 2 → Agent 3 (anomalies found)",
        },
    ]

    print("=" * 65)
    print("  Day 4 — Full LangGraph Pipeline Test")
    print("  Three agents + conditional routing")
    print("=" * 65)

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─'*65}")
        print(f"Test {i}: {tc['query']}")
        print(f"Expected path: {tc['expect_agents']}")
        print(f"{'─'*65}")

        final_state = run_pipeline(tc["query"])

        print(f"\n{'─'*65}")
        print("FINAL STATE")
        print(f"{'─'*65}")
        print(f"  Anomalies:       {len(final_state.get('anomalies', []))}")
        print(f"  has_anomalies:   {final_state.get('has_anomalies')}")
        print(f"  Recommendations: {len(final_state.get('recommendations') or [])}")

        recs = final_state.get("recommendations") or []
        for r in recs:
            print(f"    Priority {r.get('priority')}: "
                  f"[{r.get('severity','?').upper()}] {r.get('title','?')}")

        print(f"\n{'─'*65}")
        print("FINAL ANSWER (what the user sees)")
        print(f"{'─'*65}")
        answer = final_state.get("final_answer", "No answer generated.")
        # Print first 1000 chars to keep output readable
        print(answer[:1200])
        if len(answer) > 1200:
            print(f"\n  ... [{len(answer) - 1200} more chars]")

        if final_state.get("error"):
            print(f"\n  ERROR: {final_state['error']}")

    print(f"\n{'=' * 65}")
    print("  Day 4 complete. Full pipeline working end-to-end.")
    print("  Next: Day 5 — Streamlit chat UI")
    print(f"{'=' * 65}\n")