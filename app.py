"""
app.py -- Streamlit UI Frontend for the IT Help-Desk Triage & Diagnostic Agent.

Provides an interactive chat interface backed by the LangGraph state machine.
Includes custom styling, mock scenarios sidebar, database management buttons,
and a visual "Agent State Inspector" showing internal variables.
"""

import sys
import os
import uuid
import streamlit as st

# Inject src directory into system path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from db.database import initialize_database, seed_database
from agent.graph import graph
from langchain_core.messages import HumanMessage, AIMessage

# Initialize database on first launch if not already present
if not os.path.exists("helpdesk.db"):
    initialize_database()
    seed_database()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Config & Page Setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HelpBot - IT Diagnostic Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom premium styling
st.markdown(
    """
    <style>
    /* Dark professional theme overrides */
    .stApp {
        background: linear-gradient(135deg, #121824 0%, #0c0f17 100%);
        color: #e2e8f0;
    }
    
    /* Title styles */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Sidebar premium box */
    section[data-testid="stSidebar"] {
        background-color: #0c101a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Styled checklist cards */
    .checklist-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    /* Custom status pill */
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-pill.online {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .status-pill.escalated {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    /* Help instructions card */
    .scenario-btn-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Title Section
st.title("🤖 HelpBot")
st.caption("IT Triage & Diagnostic Assistant powered by LangGraph & Gemini")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls & Inspector
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ System Status")
    st.markdown(
        '<div class="checklist-card">'
        '<strong>Agent Status:</strong> <span class="status-pill online">Online</span><br>'
        f'<strong>Thread ID:</strong> <code>{st.session_state.thread_id}</code>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Database controls
    st.markdown("### 💾 Data Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset DB", use_container_width=True):
            initialize_database()
            st.toast("Database tables re-initialized.")
    with col2:
        if st.button("Seed Mock Data", use_container_width=True):
            seed_database()
            st.toast("Database populated with mock data.")

    # Conversation control
    if st.button("🔄 Reset Conversation Thread", type="primary", use_container_width=True):
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.session_state.chat_messages = []
        st.toast("Conversation thread reset.")
        st.rerun()

    # Scenario templates to make testing easy
    st.markdown("### 📝 Quick Scenarios")
    st.markdown('<p class="scenario-btn-desc">Click any to load input template:</p>', unsafe_allow_html=True)
    
    if st.button("💻 Scenario A: VPN Connection Failure", use_container_width=True):
        st.session_state.input_template = "Hi, my VPN keeps disconnecting. I can't reach the office network at all."
        st.rerun()

    if st.button("🔒 Scenario B: Account Lockout", use_container_width=True):
        st.session_state.input_template = "I can't log in. My account seems to be locked."
        st.rerun()

    if st.button("🔵 Scenario C: Blue Screen of Death", use_container_width=True):
        st.session_state.input_template = "Help, my computer just crashed and showed a blue screen with error PAGE_FAULT_IN_NONPAGED_AREA"
        st.rerun()

    # Visual Agent State Inspector
    st.markdown("### 🔍 Agent State Inspector")
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state_values = {}
    try:
        state_values = graph.get_state(config).values
    except Exception:
        pass

    if state_values:
        st.markdown(
            f"""
            <div class="checklist-card" style="font-family: monospace; font-size: 0.8rem; line-height: 1.4;">
            <strong>intent:</strong> {state_values.get("intent")}<br>
            <strong>username:</strong> {state_values.get("username")}<br>
            <strong>device_id:</strong> {state_values.get("device_id")}<br>
            <strong>missing_fields:</strong> {state_values.get("missing_fields")}<br>
            <strong>pending_action:</strong> {state_values.get("pending_action")}<br>
            <strong>pending_confirm:</strong> {state_values.get("pending_confirmation")}<br>
            <strong>ticket_id:</strong> {state_values.get("ticket_id")}<br>
            <strong>iterations:</strong> {state_values.get("iterations")}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="scenario-btn-desc">No active state yet. Start chatting below!</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────────────────────────────────────

# Display message history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Retrieve pre-filled template input if selected from sidebar
user_input = ""
if "input_template" in st.session_state:
    user_input = st.session_state.pop("input_template")

# Chat input bar
if prompt := (st.chat_input("Describe your IT issue...") or user_input):
    # Display user's input
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Diagnosing..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                )
                
                # Fetch last AI message
                last_ai = None
                for msg in reversed(result["messages"]):
                    if msg.__class__.__name__ == "AIMessage":
                        last_ai = msg.content
                        break

                if last_ai:
                    # Clean any non-ASCII characters that might display weirdly
                    clean_text = last_ai.encode("utf-8", errors="replace").decode("utf-8")
                    st.markdown(clean_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": clean_text})
                else:
                    st.markdown("I encountered an issue processing that. Please try again.")
            except Exception as e:
                st.error(f"Error executing agent node: {e}")
                
    st.rerun()
