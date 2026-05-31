"""
rag/chain.py — The RAG Chain

This is where retrieval meets generation. The chain:
    1. Takes a user question
    2. Retrieves the top-k relevant chunks from ChromaDB
    3. Formats them into a structured prompt
    4. Sends the prompt to Groq's LLM
    5. Returns a grounded, evidence-based answer

This chain becomes the backbone of Agent 1 on Day 3.
It also runs standalone for quick testing — no agents needed.

Run with:
    python -m rag.chain
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.retriever import get_retriever
from config import GROQ_API_KEY, GROQ_MODEL, TOP_K_RESULTS


# ── Prompt template ───────────────────────────────────────────────────────────
#
# This is the most important engineering decision in the RAG chain.
# The prompt tells the LLM exactly how to behave and what to do with
# the retrieved context.
#
# Key design choices made here:
#
# 1. Role definition ("You are an expert SRE...")
#    LLMs follow role framing well. Giving it a specific expert role
#    produces more focused, technical answers than a generic assistant role.
#
# 2. Strict grounding instruction ("Base your answer ONLY on the context below")
#    Without this, the LLM will mix retrieved facts with its training data,
#    which can introduce hallucinations. We want answers grounded in OUR logs.
#
# 3. Source citation instruction
#    Forces the LLM to reference which log or runbook section it used.
#    This is critical for operational tools — an engineer needs to know
#    WHERE the answer came from so they can verify it themselves.
#
# 4. Honest uncertainty instruction ("If the context does not contain...")
#    Without this, LLMs tend to make up plausible-sounding answers.
#    For incident response, a confident wrong answer is dangerous.
#
# 5. Structured response format
#    Asking for Analysis → Root Cause → Recommended Actions → Sources
#    produces a consistently structured output that is easy to scan
#    during an incident, rather than a wall of unstructured text.

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) and operations analyst.
Your job is to analyze system logs and operational data to diagnose issues and recommend solutions.

Base your answer ONLY on the context provided below — do not use outside knowledge.
If the context does not contain enough information to answer confidently, say so clearly.
Always cite which source (log file or runbook) each part of your answer comes from.

Respond in this structure:
**Analysis:** What the logs/data show about this situation.
**Root Cause:** The most likely underlying cause based on the evidence.
**Recommended Actions:** Specific steps to resolve or investigate further.
**Sources:** Which documents informed this answer."""

HUMAN_PROMPT = """Context from knowledge base:
{context}

Question: {question}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(docs) -> str:
    """
    Convert retrieved Document objects into a formatted string for the prompt.

    Why format explicitly instead of just joining page_content?
    The source metadata (filename, doc_type) is crucial context for the LLM.
    When it sees "[SOURCE: runbook_database.md | Type: runbook]" before a block
    of text, it understands this is prescriptive guidance, not observed data.
    When it sees "[SOURCE: server_logs.txt | Type: log]", it treats the text
    as factual evidence of what actually happened.
    This distinction produces noticeably better analysis.
    """
    if not docs:
        return "No relevant context found in knowledge base."

    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source   = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "unknown")
        score    = doc.metadata.get("similarity_score", "")
        score_str = f" | Relevance: {score}" if score else ""

        formatted_parts.append(
            f"[SOURCE {i}: {source} | Type: {doc_type}{score_str}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted_parts)


# ── Chain builder ─────────────────────────────────────────────────────────────

def build_rag_chain(k: int = TOP_K_RESULTS):
    """
    Assemble and return the full RAG chain.

    This uses LangChain Expression Language (LCEL) — the pipe (|) syntax.
    Each | passes the output of the left side as input to the right side.

    The chain reads as:
        {"context": retrieve+format, "question": passthrough}
        → format into prompt
        → send to LLM
        → parse text output

    RunnablePassthrough() means "take the input value and pass it through
    unchanged." We use it for "question" because the question needs to
    appear BOTH as input to the retriever AND in the final prompt.

    The lambda on "context" does two things:
        1. Gets the "question" key from the input dict
        2. Calls the retriever with it → gets Document objects
        3. Calls format_context() → turns them into a string

    The whole chain is lazy — nothing runs until you call .invoke().
    """
    retriever = get_retriever(k=k)
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,      # 0 = deterministic, no creativity
                            # For incident analysis we want facts, not poetry
    )

    chain = (
        {
            "context":  lambda x: format_context(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# ── Convenience function ──────────────────────────────────────────────────────

def ask(question: str) -> str:
    """Ask a single question and return the answer. Simple interface for agents."""
    chain = build_rag_chain()
    return chain.invoke({"question": question})


# ── Interactive test ──────────────────────────────────────────────────────────

def run_test():
    """
    Run a fixed set of questions to verify the full RAG chain works
    end-to-end: retrieval → prompt → LLM → structured answer.
    """
    test_questions = [
        "What happened with the database at 3:42 AM and what should I do about it?",
        "Why are we getting 502 errors from nginx?",
        "The disk space on /var is getting critically high. What are the immediate steps?",
    ]

    print("=" * 65)
    print("  OpsMind AI — RAG Chain Test")
    print(f"  Model: {GROQ_MODEL}")
    print("=" * 65)

    chain = build_rag_chain()

    for question in test_questions:
        print(f"\nQ: {question}")
        print("─" * 65)

        try:
            answer = chain.invoke({"question": question})
            print(answer)
        except Exception as e:
            print(f"[ERROR] {e}")

        print()

    print("=" * 65)
    print("  If answers above reference specific log lines and runbooks,")
    print("  the full RAG pipeline is working correctly.")
    print("=" * 65)


if __name__ == "__main__":
    run_test()