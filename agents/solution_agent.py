"""
agents/solution_agent.py — Agent 3: The Solution Generator

Job: Take the structured anomalies from Agent 2 and generate a complete,
     actionable response with prioritised fix steps and a final answer.

This agent receives the richest input of the three:
    - The original user query
    - The retrieved context (logs + runbooks)
    - The structured anomaly list from Agent 2

Because Agent 2 already did the hard analytical work of identifying WHAT
went wrong, Agent 3 can focus entirely on HOW to fix it — with specific
commands, priority ordering, and expected outcomes.

Key concept introduced here: MULTI-PART STRUCTURED OUTPUT
Agent 2 returned a flat list of anomaly objects.
Agent 3 returns two things:
    1. recommendations — a structured list of fix actions (machine-readable)
    2. final_answer    — a formatted string for the human (readable)

The final_answer is what appears in the Streamlit UI on Day 5.

LangGraph node contract:
    Input:  OpsState (reads: query, formatted_context, anomalies, anomaly_summary)
    Output: partial dict (writes: recommendations, final_answer)
"""

import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import OpsState
from config import GROQ_API_KEY, GROQ_MODEL


# ── Prompts ────────────────────────────────────────────────────────────────────
#
# Agent 3's prompt is the most complex of the three because it needs to
# produce TWO outputs in one LLM call: structured recommendations AND
# a human-readable final answer.
#
# We achieve this by asking the LLM to return a JSON object that contains
# BOTH: a recommendations array AND a final_answer string (with markdown).
#
# Why combine them in one call instead of two separate LLM calls?
# Cost and latency. A second LLM call doubles both. Since the final_answer
# is just a formatted rendering of the recommendations, the LLM can produce
# both simultaneously.

SOLUTION_SYSTEM_PROMPT = """You are a senior Site Reliability Engineer (SRE) providing actionable incident response guidance.

You will receive:
  1. A user's operational question
  2. Relevant log context and runbook excerpts
  3. A structured list of detected anomalies

Your job is to generate specific, prioritised fix recommendations for each anomaly.
Reference exact commands from the runbooks where available.
Prioritise by severity: critical first, then high, medium, low.

Return ONLY a valid JSON object. No markdown. No code fences. No explanation before or after.
Exact format required:
{
  "recommendations": [
    {
      "anomaly_id": "<matches id from anomaly list, e.g. anomaly_1>",
      "anomaly_type": "<type>",
      "severity": "<critical|high|medium|low>",
      "priority": 1,
      "title": "<short action title, e.g. Terminate stale DB connections>",
      "immediate_steps": [
        "<specific action with exact command if available>",
        "<next step>"
      ],
      "expected_outcome": "<what should happen after these steps>",
      "prevention": "<one sentence: how to prevent this recurring>"
    }
  ],
  "final_answer": "<A complete, well-formatted markdown response for the user. Include: summary of what happened, numbered fix steps with code blocks for commands, and a prevention note. This is what the user will read.>"
}"""

SOLUTION_HUMAN_TEMPLATE = """User question: {query}

Detected anomalies:
{anomalies_json}

Anomaly summary: {anomaly_summary}

Relevant context (logs + runbooks):
{context}

Generate prioritised fix recommendations for all detected anomalies."""


# ── JSON parsing (same robust approach as Agent 2) ────────────────────────────

def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text_stripped = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print("[Agent 3 — Solution] WARNING: Could not parse JSON output.")
    return {"recommendations": [], "final_answer": text}


# ── Agent function ─────────────────────────────────────────────────────────────

def solution_agent(state: OpsState) -> dict:
    """
    Agent 3 node function.

    Reads from state:  query, formatted_context, anomalies, anomaly_summary
    Writes to state:   recommendations, final_answer

    This is the terminal agent in the pipeline — its output is what the
    user actually sees in the Streamlit UI.
    """
    print("\n[Agent 3 — Solution] Generating fix recommendations...")

    anomalies       = state.get("anomalies", [])
    anomaly_summary = state.get("anomaly_summary", "")
    context         = state.get("formatted_context", "")
    query           = state.get("query", "")

    if not anomalies:
        print("[Agent 3 — Solution] No anomalies to generate solutions for.")
        return {
            "recommendations": [],
            "final_answer":    "No anomalies were detected in the retrieved logs.",
        }

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
        )

        # Serialise the anomaly list to JSON string for the prompt
        anomalies_json = json.dumps(anomalies, indent=2)

        messages = [
            SystemMessage(content=SOLUTION_SYSTEM_PROMPT),
            HumanMessage(content=SOLUTION_HUMAN_TEMPLATE.format(
                query=query,
                anomalies_json=anomalies_json,
                anomaly_summary=anomaly_summary,
                context=context,
            )),
        ]

        response = llm.invoke(messages)
        parsed   = extract_json(response.content)

        recommendations = parsed.get("recommendations", [])
        final_answer    = parsed.get("final_answer", "No answer generated.")

        print(f"[Agent 3 — Solution] Generated {len(recommendations)} recommendation(s).")
        for r in recommendations:
            print(f"    Priority {r.get('priority','?')}: "
                  f"[{r.get('severity','?').upper()}] {r.get('title','?')}")

        return {
            "recommendations": recommendations,
            "final_answer":    final_answer,
        }

    except Exception as e:
        print(f"[Agent 3 — Solution] ERROR: {e}")
        return {
            "recommendations": [],
            "final_answer":    f"Solution generation failed: {str(e)}",
            "error":           f"Solution agent failed: {str(e)}",
        }


# ── Direct answer (no anomalies path) ─────────────────────────────────────────

def direct_answer_agent(state: OpsState) -> dict:
    """
    Fallback node used when Agent 2 finds no anomalies.

    When the user asks a general question ("how does the connection pool work?")
    rather than an incident-specific one, Agent 2 returns has_anomalies=False.
    LangGraph routes here instead of solution_agent.

    This node uses the RAG chain directly — same as Day 2 — to produce
    a grounded answer without the anomaly analysis overhead.
    """
    print("\n[Direct Answer] No anomalies detected — answering via RAG chain.")

    from rag.chain import build_rag_chain
    try:
        chain  = build_rag_chain()
        answer = chain.invoke({"question": state["query"]})
        return {"final_answer": answer}
    except Exception as e:
        return {"final_answer": f"Could not generate answer: {str(e)}"}