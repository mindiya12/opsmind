"""
agents/analyzer_agent.py — Agent 2: The Log Analyzer

Job: Read the retrieved context, detect anomalies, return structured data.
No recommendations. No fixes. Just: "here is what went wrong, structured as JSON."

The key engineering concept introduced here is STRUCTURED LLM OUTPUT.
In Day 2, the LLM returned narrative text for a human to read.
Here, we need the LLM to return machine-readable JSON that Agent 3
can programmatically iterate over. The prompt is engineered to force
valid JSON output with no surrounding text.

Why not just parse narrative text with regex?
- Fragile: any variation in wording breaks the parser
- Unreliable: "critical error" vs "critical issue" vs "critical failure"
- Unstructured: you can't reliably extract severity, service, timestamp separately

With JSON output:
- Agent 3 can do: for anomaly in anomalies: if anomaly["severity"] == "critical"
- The data is typed, predictable, and composable across agents

LangGraph node contract:
    - Input:  OpsState (reads: formatted_context, query)
    - Output: partial dict (writes: anomalies, has_anomalies, anomaly_summary)
"""

import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import OpsState, AnomalyFlag
from config import GROQ_API_KEY, GROQ_MODEL


# ── Prompts ────────────────────────────────────────────────────────────────────
#
# The system prompt is the engineering heart of this agent.
# Key design decisions:
#
# 1. "Your ONLY job is to identify anomalies" — tight scope prevents the LLM
#    from wandering into recommendations (Agent 3's territory).
#
# 2. "Return ONLY valid JSON" — explicit instruction with no markdown, no
#    code fences, no explanation. Without this, most LLMs wrap JSON in
#    ```json ... ``` which breaks json.loads().
#
# 3. Defined anomaly_type enum — constrains the LLM to a known vocabulary.
#    Without this, Agent 3 would receive "db_pool_full", "database_exhaustion",
#    "connection_limit_reached" as three different types for the same issue.
#
# 4. Severity scale defined — "critical" means users are down RIGHT NOW.
#    Without a defined scale, "critical" and "high" become interchangeable.
#
# 5. Evidence field — forces the LLM to cite the specific log line, not just
#    describe the anomaly in its own words. Keeps the agent honest.

ANALYZER_SYSTEM_PROMPT = """You are a log analysis expert. Your ONLY job is to identify anomalies in operational data.

Analyze the context and extract factual anomalies — events that indicate system failures, degradation, threshold breaches, or errors.
Do NOT make recommendations or suggest fixes. Only identify what went wrong.

Severity definitions:
  critical — service is completely down or users cannot complete transactions RIGHT NOW
  high     — service is severely degraded, significant user impact
  medium   — warning threshold breached, trending toward failure
  low      — informational anomaly, no current user impact

Anomaly type must be one of:
  connection_pool_exhaustion | memory_pressure | disk_space_warning |
  service_failure | timeout_spike | high_error_rate | gc_pressure | other

Return ONLY a valid JSON object. No markdown. No code fences. No explanation before or after.
Exact format required:
{
  "anomalies": [
    {
      "id": "anomaly_1",
      "anomaly_type": "<type from list above>",
      "severity": "<critical|high|medium|low>",
      "service": "<affected service name>",
      "timestamp": "<timestamp from log or empty string>",
      "description": "<one sentence: what happened>",
      "evidence": "<the exact log line or passage that proves this anomaly>",
      "impact": "<what downstream services or users were affected>"
    }
  ],
  "has_anomalies": true,
  "summary": "<one sentence summarizing all anomalies>"
}

If no anomalies exist, return has_anomalies: false and an empty anomalies array."""

ANALYZER_HUMAN_TEMPLATE = """Analyze this operational context for anomalies.
Original question: {query}

Context:
{context}"""


# ── JSON parsing helper ────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM output.

    LLMs sometimes wrap JSON in markdown code fences even when told not to.
    This function handles three cases:
        1. Pure JSON (ideal case — prompt worked)
        2. JSON wrapped in ```json ... ``` (LLM ignored the instruction)
        3. JSON embedded somewhere in a longer response (partial failure)

    The regex r'\{.*\}' with re.DOTALL extracts the first { ... } block,
    which handles case 3. re.DOTALL makes . match newlines, necessary for
    multi-line JSON.

    If all parsing fails, we return a safe empty structure so the pipeline
    continues — a parse failure shouldn't crash the whole system.
    """
    text = text.strip()

    # Case 1: try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Case 2: strip markdown code fences
    text_stripped = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass

    # Case 3: extract the first {...} block from the response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # All attempts failed — return safe default
    print("[Agent 2 — Analyzer] WARNING: Could not parse LLM JSON output.")
    print(f"  Raw output (first 300 chars): {text[:300]}")
    return {"anomalies": [], "has_anomalies": False, "summary": "JSON parsing failed."}


# ── Agent function ─────────────────────────────────────────────────────────────

def analyzer_agent(state: OpsState) -> dict:
    """
    Agent 2 node function.

    Reads from state:  formatted_context, query
    Writes to state:   anomalies, has_anomalies, anomaly_summary

    Key flow:
        formatted_context → LLM with JSON prompt → parse JSON → return structured anomalies
    """
    print("\n[Agent 2 — Analyzer] Analyzing context for anomalies...")

    context = state.get("formatted_context", "")
    query   = state.get("query", "")

    if not context or context == "No relevant context found in knowledge base.":
        print("[Agent 2 — Analyzer] No context to analyze — skipping.")
        return {
            "anomalies":       [],
            "has_anomalies":   False,
            "anomaly_summary": "No context available for analysis.",
        }

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,  # Deterministic — we need consistent JSON structure
        )

        messages = [
            SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=ANALYZER_HUMAN_TEMPLATE.format(
                query=query,
                context=context,
            )),
        ]

        response  = llm.invoke(messages)
        raw_text  = response.content
        parsed    = extract_json(raw_text)

        anomalies:    list         = parsed.get("anomalies", [])
        has_anomalies: bool        = parsed.get("has_anomalies", len(anomalies) > 0)
        summary:      str          = parsed.get("summary", "")

        # Log results
        print(f"[Agent 2 — Analyzer] Detected {len(anomalies)} anomalie(s). "
              f"has_anomalies={has_anomalies}")
        for a in anomalies:
            print(f"    • [{a.get('severity','?').upper()}] "
                  f"{a.get('anomaly_type','?')} — {a.get('service','?')} "
                  f"@ {a.get('timestamp','?')}")

        return {
            "anomalies":       anomalies,
            "has_anomalies":   has_anomalies,
            "anomaly_summary": summary,
        }

    except Exception as e:
        print(f"[Agent 2 — Analyzer] ERROR: {e}")
        return {
            "anomalies":       [],
            "has_anomalies":   False,
            "anomaly_summary": f"Analysis failed: {str(e)}",
            "error":           f"Analyzer failed: {str(e)}",
        }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.retrieval_agent import retrieval_agent

    print("=" * 60)
    print("  Agent 2 — Analyzer Agent standalone test")
    print("  (runs Agent 1 first to get context, then analyzes)")
    print("=" * 60)

    # Build initial state
    state: OpsState = {
        "query":             "What anomalies are in the system logs?",
        "retrieved_docs":    [],
        "formatted_context": "",
        "anomalies":         [],
        "has_anomalies":     False,
        "anomaly_summary":   "",
        "recommendations":   None,
        "final_answer":      None,
        "error":             None,
    }

    # Run Agent 1 then merge its output into state
    state.update(retrieval_agent(state))

    # Run Agent 2
    result = analyzer_agent(state)

    print("\n" + "=" * 60)
    print("  Agent 2 output:")
    print("=" * 60)
    print(f"\nhas_anomalies: {result['has_anomalies']}")
    print(f"summary:       {result['anomaly_summary']}")
    print(f"\nAnomalies ({len(result['anomalies'])}):")
    for a in result["anomalies"]:
        print(f"\n  ID:          {a.get('id')}")
        print(f"  Type:        {a.get('anomaly_type')}")
        print(f"  Severity:    {a.get('severity')}")
        print(f"  Service:     {a.get('service')}")
        print(f"  Timestamp:   {a.get('timestamp')}")
        print(f"  Description: {a.get('description')}")
        print(f"  Evidence:    {a.get('evidence')[:100]}...")
        print(f"  Impact:      {a.get('impact')}")