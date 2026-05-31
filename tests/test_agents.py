"""
tests/test_agents.py — Individual agent tests

Tests each agent in isolation. These DO make LLM calls so they're
slower than retriever tests (~5-10 seconds each). They verify that
each agent correctly reads from state, writes expected keys, and
handles edge cases.

Run just this file:
    pytest tests/test_agents.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from graph.state    import OpsState
from agents.retrieval_agent import retrieval_agent
from agents.analyzer_agent  import analyzer_agent
from tests.conftest import make_initial_state


class TestRetrievalAgent:
    """Agent 1 — no LLM calls, fast tests."""

    def test_returns_dict(self, db_query_state):
        result = retrieval_agent(db_query_state)
        assert isinstance(result, dict)

    def test_writes_expected_keys(self, db_query_state):
        result = retrieval_agent(db_query_state)
        assert "retrieved_docs"    in result
        assert "formatted_context" in result

    def test_retrieved_docs_is_list(self, db_query_state):
        result = retrieval_agent(db_query_state)
        assert isinstance(result["retrieved_docs"], list)

    def test_returns_documents(self, db_query_state):
        result = retrieval_agent(db_query_state)
        assert len(result["retrieved_docs"]) > 0, (
            "Agent 1 returned no documents. Ensure ingestion has been run."
        )

    def test_formatted_context_is_string(self, db_query_state):
        result = retrieval_agent(db_query_state)
        assert isinstance(result["formatted_context"], str)
        assert len(result["formatted_context"]) > 0

    def test_context_contains_source_labels(self, db_query_state):
        """format_context() should label every chunk with [SOURCE N: ...]"""
        result = retrieval_agent(db_query_state)
        assert "[SOURCE" in result["formatted_context"], (
            "Formatted context missing source labels — "
            "check format_context() in rag/chain.py"
        )

    def test_handles_different_queries(self, memory_query_state, disk_query_state):
        """Agent 1 should work for any query, not just DB-related ones."""
        mem_result  = retrieval_agent(memory_query_state)
        disk_result = retrieval_agent(disk_query_state)
        assert len(mem_result["retrieved_docs"])  > 0
        assert len(disk_result["retrieved_docs"]) > 0

    def test_does_not_write_unexpected_keys(self, db_query_state):
        """Agent 1 should only write its designated state keys."""
        result = retrieval_agent(db_query_state)
        allowed_keys = {"retrieved_docs", "formatted_context", "error"}
        unexpected   = set(result.keys()) - allowed_keys
        assert not unexpected, f"Agent 1 wrote unexpected keys: {unexpected}"


class TestAnalyzerAgent:
    """Agent 2 — makes LLM calls, tests structured JSON output."""

    def _run_both_agents(self, state: OpsState) -> dict:
        """Helper: run Agent 1 then Agent 2, return Agent 2's output."""
        state.update(retrieval_agent(state))
        return analyzer_agent(state)

    def test_returns_dict(self, db_query_state):
        result = self._run_both_agents(db_query_state)
        assert isinstance(result, dict)

    def test_writes_expected_keys(self, db_query_state):
        result = self._run_both_agents(db_query_state)
        assert "anomalies"       in result
        assert "has_anomalies"   in result
        assert "anomaly_summary" in result

    def test_anomalies_is_list(self, db_query_state):
        result = self._run_both_agents(db_query_state)
        assert isinstance(result["anomalies"], list)

    def test_has_anomalies_is_bool(self, db_query_state):
        result = self._run_both_agents(db_query_state)
        assert isinstance(result["has_anomalies"], bool)

    def test_detects_db_anomalies(self, db_query_state):
        """Agent 2 should detect at least one anomaly in DB incident logs."""
        result = self._run_both_agents(db_query_state)
        assert result["has_anomalies"] is True, (
            "Agent 2 found no anomalies for a DB incident query. "
            "Check the analyzer prompt or retrieval quality."
        )

    def test_anomaly_objects_have_required_fields(self, db_query_state):
        """Each anomaly dict must have all required fields."""
        result   = self._run_both_agents(db_query_state)
        required = {"id", "anomaly_type", "severity", "service",
                    "timestamp", "description", "evidence", "impact"}
        for anomaly in result["anomalies"]:
            missing = required - set(anomaly.keys())
            assert not missing, (
                f"Anomaly missing fields: {missing}. Got: {list(anomaly.keys())}"
            )

    def test_anomaly_severity_is_valid(self, db_query_state):
        """Severity must be one of the four defined levels."""
        result      = self._run_both_agents(db_query_state)
        valid_sevs  = {"critical", "high", "medium", "low"}
        for anomaly in result["anomalies"]:
            sev = anomaly.get("severity", "").lower()
            assert sev in valid_sevs, (
                f"Invalid severity '{sev}'. Must be one of {valid_sevs}"
            )

    def test_anomaly_type_is_valid(self, db_query_state):
        """Anomaly type must be from the defined enum in the prompt."""
        result     = self._run_both_agents(db_query_state)
        valid_types = {
            "connection_pool_exhaustion", "memory_pressure", "disk_space_warning",
            "service_failure", "timeout_spike", "high_error_rate",
            "gc_pressure", "other"
        }
        for anomaly in result["anomalies"]:
            atype = anomaly.get("anomaly_type", "")
            assert atype in valid_types, (
                f"Unknown anomaly_type '{atype}'. "
                f"LLM may have deviated from the prompt enum."
            )

    def test_summary_is_non_empty_string(self, db_query_state):
        result = self._run_both_agents(db_query_state)
        summary = result.get("anomaly_summary", "")
        assert isinstance(summary, str) and len(summary) > 0

    def test_has_anomalies_consistent_with_list(self, db_query_state):
        """has_anomalies=True should mean anomalies list is non-empty."""
        result = self._run_both_agents(db_query_state)
        if result["has_anomalies"]:
            assert len(result["anomalies"]) > 0, (
                "has_anomalies=True but anomalies list is empty — inconsistent state"
            )

    def test_handles_empty_context(self):
        """Agent 2 should handle missing context gracefully."""
        state = make_initial_state("test query")
        state["formatted_context"] = ""
        result = analyzer_agent(state)
        assert result["has_anomalies"]   is False
        assert result["anomalies"]       == []
        assert isinstance(result["anomaly_summary"], str)