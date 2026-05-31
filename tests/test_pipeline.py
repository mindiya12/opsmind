"""
tests/test_pipeline.py — End-to-end pipeline tests

These are the highest-level tests — they exercise the full LangGraph
pipeline from a user query all the way to a final answer.
They're the slowest (~15-20 seconds each) but the most valuable for
demonstrating the system works completely.

Run just this file:
    pytest tests/test_pipeline.py -v

Run all tests:
    pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from graph.pipeline import run_pipeline


class TestPipelineStructure:
    """Verify the compiled pipeline has the correct shape."""

    def test_pipeline_compiles(self, pipeline):
        """Pipeline should compile without errors."""
        assert pipeline is not None

    def test_pipeline_has_invoke_method(self, pipeline):
        """CompiledGraph must be callable via .invoke()."""
        assert hasattr(pipeline, "invoke"), (
            "Pipeline missing .invoke() — compilation may have failed"
        )


class TestPipelineEndToEnd:
    """Full pipeline tests — run all three agents and verify final state."""

    def test_db_incident_full_pipeline(self):
        """DB incident query should produce anomalies and fix recommendations."""
        state = run_pipeline(
            "What happened with the database connection pool at 3:42 AM?"
        )

        # Agent 1 output
        assert len(state["retrieved_docs"])    > 0,  "Agent 1 returned no docs"
        assert len(state["formatted_context"]) > 0,  "Agent 1 produced no context"

        # Agent 2 output
        assert state["has_anomalies"]          is True, "Agent 2 found no anomalies for DB incident"
        assert len(state["anomalies"])         > 0,  "Anomaly list is empty"

        # Agent 3 output
        assert state["final_answer"]           is not None, "No final answer generated"
        assert len(state["final_answer"])      > 100,       "Final answer too short"

    def test_disk_incident_full_pipeline(self):
        """Disk space query should detect disk_space_warning anomalies."""
        state = run_pipeline(
            "Are there any disk space issues on the /var partition?"
        )

        assert state["has_anomalies"] is True
        anomaly_types = [a.get("anomaly_type") for a in state["anomalies"]]
        assert "disk_space_warning" in anomaly_types, (
            f"Expected disk_space_warning. Got: {anomaly_types}"
        )

    def test_memory_incident_full_pipeline(self):
        """Memory query should detect memory_pressure or gc_pressure."""
        state = run_pipeline(
            "Are there any JVM memory or garbage collection issues?"
        )

        assert state["has_anomalies"] is True
        anomaly_types = [a.get("anomaly_type") for a in state["anomalies"]]
        memory_types  = {"memory_pressure", "gc_pressure"}
        found         = memory_types & set(anomaly_types)
        assert found, (
            f"Expected memory/GC anomaly types. Got: {anomaly_types}"
        )

    def test_final_answer_is_string(self):
        """final_answer must always be a non-empty string."""
        state = run_pipeline("What anomalies are in the logs?")
        assert isinstance(state["final_answer"], str)
        assert len(state["final_answer"]) > 0

    def test_state_keys_all_populated(self):
        """After pipeline runs, no expected key should be None or missing."""
        state = run_pipeline("What went wrong overnight?")
        required_keys = [
            "query", "retrieved_docs", "formatted_context",
            "anomalies", "has_anomalies", "anomaly_summary", "final_answer"
        ]
        for key in required_keys:
            assert key in state,          f"Key '{key}' missing from final state"
            assert state[key] is not None, f"Key '{key}' is None in final state"

    def test_recommendations_when_anomalies_found(self):
        """When anomalies are detected, recommendations should be populated."""
        state = run_pipeline(
            "What happened with the database and what should I do?"
        )
        if state["has_anomalies"]:
            recs = state.get("recommendations") or []
            assert len(recs) > 0, (
                "Anomalies detected but recommendations list is empty — "
                "Agent 3 may have failed silently"
            )

    def test_recommendation_structure(self):
        """Each recommendation must have priority, title, and immediate_steps."""
        state = run_pipeline("Database connection pool issues — how to fix?")
        recs  = state.get("recommendations") or []

        for rec in recs:
            assert "priority"        in rec, f"Recommendation missing 'priority': {rec}"
            assert "title"           in rec, f"Recommendation missing 'title': {rec}"
            assert "immediate_steps" in rec, f"Recommendation missing 'immediate_steps': {rec}"
            assert isinstance(rec["immediate_steps"], list)
            assert len(rec["immediate_steps"]) > 0

    def test_no_error_in_normal_operation(self):
        """Normal queries should not populate the error field."""
        state = run_pipeline("What are the main incidents in the logs?")
        assert state.get("error") is None, (
            f"Pipeline set error field unexpectedly: {state.get('error')}"
        )


class TestPipelineRouting:
    """Verify conditional routing works correctly."""

    def test_anomaly_path_populates_recommendations(self):
        """When anomalies found, Agent 3 runs and recommendations are set."""
        state = run_pipeline("Database connection pool exhausted — critical incident")
        if state["has_anomalies"]:
            # Agent 3 ran — recommendations should be populated
            assert state.get("recommendations") is not None, (
                "has_anomalies=True but recommendations is None — "
                "Agent 3 may not have been routed to"
            )

    def test_final_answer_always_present(self):
        """Both routing paths must produce a final_answer."""
        state = run_pipeline("Tell me about the system logs")
        assert state.get("final_answer") is not None
        assert len(state["final_answer"]) > 50