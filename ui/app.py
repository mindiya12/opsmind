"""
ui/app.py — OpsMind AI Chat Interface

Streamlit-based chat UI that connects directly to the LangGraph pipeline.
Every user message triggers the full three-agent pipeline and renders
the structured result in a chat bubble with source citations.

Run with:
    streamlit run ui/app.py
    (from the opsmind_ai/ directory)

Key Streamlit patterns used:
    st.session_state   — persists data across reruns
    @st.cache_resource — caches the compiled pipeline (built once, reused)
    st.chat_message()  — renders user/assistant chat bubbles
    st.chat_input()    — the message input box at the bottom
    st.sidebar         — the source panel on the left
    st.spinner()       — loading indicator while pipeline runs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from graph.pipeline import build_pipeline
from graph.state    import OpsState


# ── Page configuration ─────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.

st.set_page_config(
    page_title="OpsMind AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Anomaly severity badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px 3px 2px 0;
    }
    .badge-critical { background: #FCEBEB; color: #A32D2D; }
    .badge-high     { background: #FAEEDA; color: #854F0B; }
    .badge-medium   { background: #E6F1FB; color: #185FA5; }
    .badge-low      { background: #EAF3DE; color: #3B6D11; }

    /* Source chunk cards in sidebar */
    .source-card {
        background: rgba(127,119,221,0.07);
        border-left: 3px solid #7F77DD;
        border-radius: 0 6px 6px 0;
        padding: 8px 10px;
        margin-bottom: 8px;
        font-size: 12px;
    }
    .source-card-runbook {
        border-left-color: #1D9E75;
        background: rgba(29,158,117,0.07);
    }
    .score-pill {
        display: inline-block;
        background: #EEEDFE;
        color: #534AB7;
        border-radius: 99px;
        padding: 1px 7px;
        font-size: 10px;
        font-weight: 600;
        float: right;
    }
    .score-pill-runbook {
        background: #E1F5EE;
        color: #0F6E56;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(127,119,221,0.1), rgba(29,158,117,0.08));
        border: 1px solid rgba(127,119,221,0.2);
        border-radius: 12px;
        padding: 24px 28px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache the compiled pipeline ────────────────────────────────────────────────
# @st.cache_resource runs build_pipeline() ONCE and reuses the compiled graph
# on every subsequent rerun. Without this, the graph would recompile on every
# message, adding ~1-2 seconds of unnecessary overhead.

@st.cache_resource
def get_pipeline():
    return build_pipeline()


# ── Session state initialisation ───────────────────────────────────────────────
# These keys are set once on first load and then mutated in place.
# Streamlit preserves them across reruns within the same browser session.

if "messages" not in st.session_state:
    st.session_state.messages = []         # List of {role, content, meta}

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []     # Retrieved docs from last query

if "last_anomalies" not in st.session_state:
    st.session_state.last_anomalies = []   # Anomalies from last query

if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = []


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 OpsMind AI")
    st.markdown("Multi-agent operational intelligence powered by LangGraph + RAG")
    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages           = []
        st.session_state.last_sources       = []
        st.session_state.last_anomalies     = []
        st.session_state.last_recommendations = []
        st.rerun()

    st.divider()

    # Anomaly panel — shows badges for last query's anomalies
    if st.session_state.last_anomalies:
        st.markdown("**⚠️ Detected anomalies**")
        for a in st.session_state.last_anomalies:
            sev  = a.get("severity", "low")
            typ  = a.get("anomaly_type", "unknown").replace("_", " ")
            svc  = a.get("service", "")
            ts   = a.get("timestamp", "")
            ts_str = f" · {ts}" if ts else ""
            st.markdown(
                f'<span class="badge badge-{sev}">{sev.upper()}</span> '
                f'**{typ}** in `{svc}`{ts_str}',
                unsafe_allow_html=True
            )
        st.divider()

    # Source panel — shows retrieved chunks with relevance scores
    if st.session_state.last_sources:
        st.markdown("**📚 Retrieved sources**")
        st.caption("Chunks used to answer your last question")

        for i, doc in enumerate(st.session_state.last_sources, 1):
            source   = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("doc_type", "log")
            score    = doc.metadata.get("similarity_score", "")
            preview  = doc.page_content.strip()[:140].replace("\n", " ")
            score_str = f"{score}" if score else ""

            card_class  = "source-card-runbook" if doc_type == "runbook" else ""
            score_class = "score-pill-runbook"   if doc_type == "runbook" else ""
            icon        = "📖" if doc_type == "runbook" else "📋"

            st.markdown(
                f'<div class="source-card {card_class}">'
                f'<span class="score-pill {score_class}">{score_str}</span>'
                f'{icon} <strong>{source}</strong><br>'
                f'<span style="color:#888;font-size:11px">{preview}...</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("Source chunks will appear here after your first question.")

    st.divider()
    st.caption("Agents: Retrieval → Analyzer → Solution Generator")
    st.caption("Model: llama-3.3-70b-versatile via Groq")


# ── Main chat area ─────────────────────────────────────────────────────────────

# Header
st.markdown("# OpsMind AI")
st.markdown("Ask anything about your system logs, incidents, or operational issues.")

# Welcome card — only shown when chat is empty
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
    <h4 style="margin-top:0">👋 What can I help you diagnose?</h4>
    <p style="margin-bottom:8px;color:#555">Try asking:</p>
    <ul style="color:#555;margin:0">
        <li>What happened with the database at 3:42 AM?</li>
        <li>Are there any memory or disk space issues?</li>
        <li>Why are we getting 502 errors from nginx?</li>
        <li>What anomalies occurred overnight?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show anomaly summary badge on assistant messages that found anomalies
        if msg["role"] == "assistant" and msg.get("meta", {}).get("anomaly_count", 0) > 0:
            count = msg["meta"]["anomaly_count"]
            st.caption(f"⚠️ {count} anomaly/anomalies detected and addressed above")


# ── Chat input and pipeline execution ─────────────────────────────────────────

if prompt := st.chat_input("Ask about your logs or incidents..."):

    # 1. Render user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Run the full pipeline with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Agents running — retrieving context, analyzing logs, generating fix..."):
            try:
                pipeline = get_pipeline()

                initial_state = {
                    "query":             prompt,
                    "retrieved_docs":    [],
                    "formatted_context": "",
                    "anomalies":         [],
                    "has_anomalies":     False,
                    "anomaly_summary":   "",
                    "recommendations":   None,
                    "final_answer":      None,
                    "error":             None,
                }

                final_state: OpsState = pipeline.invoke(initial_state)

                # Extract outputs
                answer          = final_state.get("final_answer") or "No answer generated."
                sources         = final_state.get("retrieved_docs", [])
                anomalies       = final_state.get("anomalies", [])
                recommendations = final_state.get("recommendations") or []
                error           = final_state.get("error")

                # Update sidebar state
                st.session_state.last_sources         = sources
                st.session_state.last_anomalies       = anomalies
                st.session_state.last_recommendations = recommendations

                # 3. Render the answer
                if error:
                    st.warning(f"⚠️ Pipeline encountered an issue: {error}")

                st.markdown(answer)

                # Show anomaly count if any found
                if anomalies:
                    st.caption(f"⚠️ {len(anomalies)} anomaly/anomalies detected and addressed above")

                # 4. Save assistant message to history
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "meta":    {"anomaly_count": len(anomalies)},
                })

                # 5. Rerun to refresh the sidebar with new sources + anomalies
                st.rerun()

            except Exception as e:
                error_msg = f"Pipeline error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": f"❌ {error_msg}",
                    "meta":    {},
                })