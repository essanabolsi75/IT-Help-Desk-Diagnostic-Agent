"""
app.py -- Streamlit UI for the IT Help-Desk Triage & Diagnostic Agent.

Features:
  - Premium dark-mode chat interface with custom CSS
  - Left sidebar: Agent State Inspector with live memory fields
  - Main panel: Multi-turn conversational chat with the LangGraph agent
  - Quick-scenario template buttons for rapid testing
  - Database reset/seed controls
"""

import sys
import os
import uuid
import streamlit as st

# ---------------------------------------------------------------------------
# Path & DB setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from db.database import initialize_database, seed_database, DB_PATH
from agent.graph import graph
from langchain_core.messages import HumanMessage

# Auto-init DB on first launch
if not os.path.exists(DB_PATH):
    initialize_database()
    seed_database()

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HelpBot - IT Diagnostic Agent",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS -- premium glassmorphic dark-mode styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global overrides */
.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* Hide default Streamlit header/footer for cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Sidebar styling ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e18 0%, #0c1120 100%) !important;
    border-right: 1px solid rgba(16, 185, 129, 0.15);
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #10b981 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid rgba(16, 185, 129, 0.15);
    padding-bottom: 6px;
}

/* ── Glass cards ────────────────────────────────────────────────── */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    transition: border-color 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(16, 185, 129, 0.4);
}

/* ── Status pills ───────────────────────────────────────────────── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.pill-green {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.pill-yellow {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.pill-red {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.pill-blue {
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    border: 1px solid rgba(59, 130, 246, 0.4);
}
.pill-gray {
    background: rgba(100, 116, 139, 0.15);
    color: #94a3b8;
    border: 1px solid rgba(100, 116, 139, 0.4);
}

/* ── State field rows ───────────────────────────────────────────── */
.state-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    font-size: 0.82rem;
    border-bottom: 1px solid rgba(51, 65, 85, 0.25);
}
.state-row:last-child { border-bottom: none; }
.state-label {
    color: #94a3b8;
    font-weight: 500;
}
.state-value {
    color: #e2e8f0;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.78rem;
    max-width: 180px;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Missing fields checklist ───────────────────────────────────── */
.field-check {
    font-size: 0.8rem;
    padding: 2px 0;
}
.field-done { color: #10b981; }
.field-missing { color: #f59e0b; }

/* ── Title area ─────────────────────────────────────────────────── */
.title-block {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.title-block h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #10b981, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    padding: 0 !important;
}
.title-caption {
    color: #64748b;
    font-size: 0.85rem;
    margin-bottom: 20px;
}

/* ── Scenario buttons ───────────────────────────────────────────── */
.scenario-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Helper: get current graph state safely
# ---------------------------------------------------------------------------
def get_agent_state() -> dict:
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        return graph.get_state(config).values or {}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Helper: render a state field row
# ---------------------------------------------------------------------------
def state_row(label: str, value, pill_class: str = ""):
    val_str = str(value) if value is not None else "---"
    if pill_class and value is not None:
        val_html = f'<span class="pill {pill_class}">{val_str}</span>'
    else:
        val_html = f'<span class="state-value">{val_str}</span>'
    return f'<div class="state-row"><span class="state-label">{label}</span>{val_html}</div>'

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo / branding area
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 6px 0;">
        <span style="font-size:2rem;">🛠️</span>
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:4px;">HelpBot</div>
        <div style="font-size:0.7rem; color:#64748b; letter-spacing:0.05em;">IT DIAGNOSTIC AGENT</div>
    </div>
    """, unsafe_allow_html=True)

    # ── System Status ──
    st.markdown("### System Status")
    st.markdown(f"""
    <div class="glass-card">
        {state_row("Agent", "ONLINE", "pill-green")}
        {state_row("Thread ID", st.session_state.thread_id)}
        {state_row("Model", "gemini-3.1-flash-lite")}
    </div>
    """, unsafe_allow_html=True)

    # ── Agent Memory Inspector ──
    st.markdown("### Agent Memory")
    sv = get_agent_state()

    intent = sv.get("intent")
    intent_pill = "pill-blue" if intent and intent != "unknown" else "pill-gray"

    confirm = sv.get("pending_confirmation")
    confirm_pill = "pill-yellow" if confirm else "pill-gray"

    ticket = sv.get("ticket_id")
    ticket_pill = "pill-green" if ticket else "pill-gray"

    st.markdown(f"""
    <div class="glass-card">
        {state_row("Intent", intent or "---", intent_pill)}
        {state_row("Username", sv.get("username"))}
        {state_row("Device ID", sv.get("device_id"))}
        {state_row("Iterations", sv.get("iterations"))}
        {state_row("Awaiting Confirm", confirm, confirm_pill)}
        {state_row("Ticket ID", f"#{ticket}" if ticket else "---", ticket_pill)}
    </div>
    """, unsafe_allow_html=True)

    # ── Diagnostic Checklist ──
    st.markdown("### Diagnostic Checklist")
    missing = sv.get("missing_fields") or []
    gathered = sv.get("gathered_info") or {}

    # Define all possible fields for display
    all_fields = {
        "username": "Employee Username",
        "ping_result": "Ping / Network Test",
        "error_code": "Error / Stop Code",
        "disk_space": "Disk Space Info",
    }
    checklist_html = '<div class="glass-card">'
    has_any = False
    for field_key, field_label in all_fields.items():
        if field_key in missing:
            checklist_html += f'<div class="field-check field-missing">○ {field_label}</div>'
            has_any = True
        elif field_key in gathered or (field_key == "username" and sv.get("username")):
            checklist_html += f'<div class="field-check field-done">● {field_label}</div>'
            has_any = True
    if not has_any:
        checklist_html += '<div style="font-size:0.78rem; color:#475569;">No active checklist</div>'
    checklist_html += '</div>'
    st.markdown(checklist_html, unsafe_allow_html=True)

    # ── Controls ──
    st.markdown("### Controls")
    if st.button("🔄 New Conversation", type="primary", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset Database", use_container_width=True):
        initialize_database()
        seed_database()
        st.toast("Database reset to default mock values.")

    # ── Quick Scenarios ──
    st.markdown("### Quick Scenarios")
    st.markdown('<div class="scenario-label">Click to auto-fill a test message</div>', unsafe_allow_html=True)

    scenarios = [
        ("🌐 VPN Disconnect", "Hi, my VPN keeps disconnecting. I can't reach the office network at all."),
        ("🔒 Account Lockout", "I can't log in. My account seems to be locked."),
        ("💀 Blue Screen (BSOD)", "My computer crashed with a blue screen showing PAGE_FAULT_IN_NONPAGED_AREA"),
        ("🐌 Slow Wi-Fi", "My internet is extremely slow today, pages take forever to load."),
        ("🔑 MFA Reset", "I got a new phone and need to reset my two-factor authentication."),
    ]
    for label, msg in scenarios:
        if st.button(label, use_container_width=True):
            st.session_state.pending_input = msg
            st.rerun()

# ---------------------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------------------
# Title
st.markdown("""
<div class="title-block">
    <h1>HelpBot</h1>
</div>
<div class="title-caption">Interactive IT Triage & Diagnostic Assistant &mdash; powered by LangGraph &amp; Gemini</div>
""", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🛠️" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# Check for pending scenario input
pending = st.session_state.pop("pending_input", None)

# Chat input
if prompt := (st.chat_input("Describe your IT issue...") or pending):
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant", avatar="🛠️"):
        with st.spinner("Diagnosing..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                )

                # Extract last AI message
                last_ai = None
                for msg in reversed(result["messages"]):
                    if msg.__class__.__name__ == "AIMessage":
                        last_ai = msg.content
                        break

                response_text = last_ai or "I encountered an issue processing your request. Please try again."
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"**Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    st.rerun()
