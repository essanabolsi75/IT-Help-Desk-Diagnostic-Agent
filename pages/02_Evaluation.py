"""
pages/02_Evaluation.py — Automated Evaluation Dashboard for the IT Help-Desk Agent.

Runs 20 scripted test conversations through the live LangGraph agent,
evaluates pass/fail against expected outcomes, and displays a summary report.

Lifecycle:
  1. Reseed DB to defaults before the suite starts.
  2. Reseed DB before each individual test for isolation.
  3. Reseed DB after the full suite completes.
"""

import sys
import os
import uuid
import time
import csv
import io

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — same pattern as app.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from db.database import initialize_database, seed_database, get_db_connection
from agent.graph import graph
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Evaluation Dashboard — HelpBot",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling — matches the main app theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif !important; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Summary metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
    border: 1px solid rgba(100,116,139,0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-number {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.pass-color  { color: #22c55e; }
.fail-color  { color: #ef4444; }
.total-color { color: #60a5fa; }
.rate-color  { color: #f59e0b; }

/* Result row badges */
.badge-pass { background:#166534; color:#bbf7d0; padding:2px 10px; border-radius:99px; font-size:0.75rem; font-weight:600; }
.badge-fail { background:#7f1d1d; color:#fecaca; padding:2px 10px; border-radius:99px; font-size:0.75rem; font-weight:600; }
.badge-cat  { background:#1e3a5f; color:#93c5fd; padding:2px 10px; border-radius:99px; font-size:0.72rem; }

/* Trace block */
.trace-user  { background:rgba(30,41,59,0.7); border-left:3px solid #3b82f6; padding:8px 12px; border-radius:0 6px 6px 0; margin:4px 0; font-size:0.85rem; }
.trace-agent { background:rgba(15,23,42,0.8); border-left:3px solid #22c55e; padding:8px 12px; border-radius:0 6px 6px 0; margin:4px 0; font-size:0.85rem; }
.trace-label { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 20 Test Scenarios
# ---------------------------------------------------------------------------
TEST_SUITE = [
    # ── 1-2: Chitchat / Unknown Intent ──────────────────────────────────────
    {
        "id": 1, "name": "Greeting Only (Chitchat)", "category": "chitchat",
        "turns": ["Hello! How are you doing today?"],
        "report_expected": False, "expected_status": None,
        "description": "Pure greeting with no IT issue. Agent should ask what's wrong.",
    },
    {
        "id": 2, "name": "Off-Topic Question", "category": "chitchat",
        "turns": ["What is the weather like in New York today?"],
        "report_expected": False, "expected_status": None,
        "description": "Non-IT question. Agent should stay scoped and redirect.",
    },
    # ── 3-4: Incomplete Info / Gather ───────────────────────────────────────
    {
        "id": 3, "name": "Incomplete Info — Single Turn (Gather)", "category": "gather_info",
        "turns": ["I have a VPN issue"],
        "report_expected": False, "expected_status": None,
        "description": "User provides only intent with no username or ping. Agent should ask for more info.",
    },
    {
        "id": 4, "name": "Invalid Username — Agent Asks Again", "category": "gather_info",
        "turns": ["My account is locked.", "notarealusername123"],
        "report_expected": False, "expected_status": None,
        "description": "User supplies a username not in the DB. Agent must inform them and re-ask.",
    },
    # ── 5-8: Account Lockout ─────────────────────────────────────────────────
    {
        "id": 5, "name": "Account Lockout — Unlock Confirmed → Resolved", "category": "account_lockout",
        "turns": [
            "My account is locked, I cannot log in.",
            "bob_jones",
            "yes please unlock it",
            "yes it is working now",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "Full happy-path: locked account, unlock confirmed, user confirms resolution.",
    },
    {
        "id": 6, "name": "Account Lockout — Unlock Denied → Resolved", "category": "account_lockout",
        "turns": [
            "I cannot login, my account seems to be locked.",
            "bob_jones",
            "no, cancel the unlock",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "User denies the unlock. Ticket should be resolved/closed as cancelled.",
    },
    {
        "id": 7, "name": "Account Lockout — Wrong Username Then Correct", "category": "account_lockout",
        "turns": [
            "My account is locked.",
            "wronguser999",
            "bob_jones",
            "yes unlock it",
            "yes fixed",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "User first gives invalid username, then corrects it. Tests error-recovery flow.",
    },
    {
        "id": 8, "name": "Account Lockout — Username Provided Upfront", "category": "account_lockout",
        "turns": [
            "My account bob_jones is locked, I cannot get in.",
            "yes please unlock",
            "yes it works",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "Username extracted from first message. Tests upfront-extraction bug fix.",
    },
    # ── 9-11: VPN Issues ──────────────────────────────────────────────────────
    {
        "id": 9, "name": "VPN — 100% Packet Loss → Escalated", "category": "vpn",
        "turns": [
            "My VPN keeps disconnecting, I cannot reach the office network.",
            "alice_smith",
            "Packets: Sent = 4, Received = 0, Lost = 4 (100% loss)",
            "no still not working",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "Complete network failure (100% loss). Tips won't fix it — should escalate.",
    },
    {
        "id": 10, "name": "VPN — Healthy Ping → Tips → Resolved", "category": "vpn",
        "turns": [
            "VPN is not connecting for me today.",
            "alice_smith",
            "Packets: Sent = 4, Received = 4, Lost = 0 (0% loss)",
            "yes that fixed it",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "Network is actually healthy. Tips (restart VPN client) resolve it.",
    },
    {
        "id": 11, "name": "VPN — Username in First Message", "category": "vpn",
        "turns": [
            "Hi, alice_smith here. My VPN disconnects every 10 minutes.",
            "Packets: Sent = 4, Received = 2, Lost = 2 (50% loss)",
            "no its still dropping",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "Username extracted from first message. 50% loss — escalated.",
    },
    # ── 12-13: Slow Wi-Fi ──────────────────────────────────────────────────────
    {
        "id": 12, "name": "Slow Wi-Fi — Tips Resolve Issue", "category": "slow_wifi",
        "turns": [
            "My internet is very slow today, pages take forever to load.",
            "Packets: Sent = 4, Received = 3, Lost = 1 (25% loss)",
            "yes it is faster now",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "Partial packet loss, DNS/IP tips resolve it.",
    },
    {
        "id": 13, "name": "Slow Wi-Fi — Issue Persists", "category": "slow_wifi",
        "turns": [
            "My wifi is extremely slow and keeps dropping.",
            "Packets: Sent = 4, Received = 0, Lost = 4 (100% loss)",
            "no still slow",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "Complete packet loss, tips don't resolve — escalated.",
    },
    # ── 14-17: BSOD ────────────────────────────────────────────────────────────
    {
        "id": 14, "name": "BSOD — Error Code Upfront → Tips → Resolved", "category": "bsod",
        "turns": [
            "My computer crashed with a blue screen PAGE_FAULT_IN_NONPAGED_AREA. Username is charlie_brown.",
            "yes it is fixed",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "Error code and username given in first message. Tests upfront extraction. Resolved.",
    },
    {
        "id": 15, "name": "BSOD — Error Code Upfront → Tips → Persists", "category": "bsod",
        "turns": [
            "BSOD with CRITICAL_PROCESS_DIED. I am charlie_brown.",
            "no still crashing",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "All info in first message, tips don't help — escalated to hardware team.",
    },
    {
        "id": 16, "name": "BSOD — Ambiguous Reply ('yes I saw it')", "category": "bsod",
        "turns": [
            "Blue screen of death, stop code PAGE_FAULT_IN_NONPAGED_AREA. I am charlie_brown.",
            "yes i saw that error",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "Tests that 'yes I saw it' is not mistaken for 'yes it is fixed'. Should escalate.",
    },
    {
        "id": 17, "name": "BSOD — Multi-Turn Info Gathering", "category": "bsod",
        "turns": [
            "My laptop keeps crashing with a blue screen.",
            "charlie_brown",
            "SYSTEM_THREAD_EXCEPTION_NOT_HANDLED",
            "no still happening",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "Username and error code gathered across separate turns. Full multi-turn flow.",
    },
    # ── 18: MFA / Password ─────────────────────────────────────────────────────
    {
        "id": 18, "name": "MFA Reset — Resolved", "category": "mfa_reset",
        "turns": [
            "I got a new phone and need to reset my two-factor authentication.",
            "diana_prince",
            "yes that resolved it",
        ],
        "report_expected": True, "expected_status": "Resolved",
        "description": "MFA reset flow. Tips guide user through re-enrollment.",
    },
    # ── 19: Software Update ────────────────────────────────────────────────────
    {
        "id": 19, "name": "Software Update Error — Full Flow", "category": "update_error",
        "turns": [
            "Windows update keeps failing on my machine.",
            "charlie_brown",
            "only 2GB free on my C: drive",
            "error code 0x80244007",
            "no still failing",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "Full update error flow with disk space and error code. Low disk → escalated.",
    },
    # ── 20: Multi-Issue Same Thread ────────────────────────────────────────────
    {
        "id": 20, "name": "Multi-Issue: Lockout Then VPN (Same Thread)", "category": "multi",
        "turns": [
            "My account is locked. Username is bob_jones.",
            "yes unlock it",
            "yes it works now",
            "Now I have a different problem — my VPN keeps disconnecting.",
            "Packets: Sent = 4, Received = 0, Lost = 4 (100% loss)",
            "no still broken",
        ],
        "report_expected": True, "expected_status": "Escalated",
        "description": "Two separate issues in one thread. Tests state reset between issues.",
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_single_test(test: dict) -> dict:
    """
    Seeds the DB, runs all conversation turns through the graph,
    reads the final state, and returns a structured result dict.
    """
    # Reseed for isolation before every test
    try:
        initialize_database()
        seed_database()
    except Exception as e:
        pass  # If seeding fails, still try to run the test

    thread_id = f"eval-{test['id']}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    turns_trace = []
    error_msg = None

    try:
        for user_msg in test["turns"]:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config,
            )
            last_ai = ""
            for msg in reversed(result.get("messages", [])):
                if msg.__class__.__name__ == "AIMessage":
                    last_ai = msg.content
                    break
            turns_trace.append({"user": user_msg, "agent": last_ai})

        # Read final state
        final_state = graph.get_state(config).values or {}
        final_report = final_state.get("final_report")
        intent = final_state.get("intent")
        missing = final_state.get("missing_fields")

    except Exception as e:
        error_msg = str(e)
        final_report = None
        intent = None
        missing = None

    # ── Evaluate pass / fail ──────────────────────────────────────────────────
    actual_status = None
    passed = False

    if error_msg:
        passed = False
        actual_status = "ERROR"
    elif test["report_expected"]:
        if final_report is None:
            passed = False
            actual_status = "No Report"
        else:
            # Query the SQLite DB directly for the final resolution status of this ticket
            ticket_id = final_state.get("ticket_id")
            if ticket_id:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT resolution_status FROM tickets WHERE ticket_id = ?;", (ticket_id,))
                    row = cursor.fetchone()
                    conn.close()
                    if row:
                        actual_status = row["resolution_status"]  # "Resolved" or "Escalated"
                    else:
                        actual_status = "Report (Ticket not found in DB)"
                except Exception as e:
                    actual_status = f"Report (DB Error: {str(e)})"
            else:
                # Fallback to strict report checking if ticket_id is missing
                if "**Status**: **RESOLVED**" in final_report or "Status: **RESOLVED**" in final_report:
                    actual_status = "Resolved"
                elif "**Status**: **ESCALATED**" in final_report or "Status: **ESCALATED**" in final_report:
                    actual_status = "Escalated"
                else:
                    actual_status = "Report (Unknown status)"
            passed = (test["expected_status"] is None) or (actual_status == test["expected_status"])
    else:
        # No report expected
        passed = final_report is None
        actual_status = "No Report (expected)" if passed else f"Report generated (unexpected)"

    return {
        "id":              test["id"],
        "name":            test["name"],
        "category":        test["category"],
        "description":     test["description"],
        "turns_count":     len(test["turns"]),
        "expected_status": test["expected_status"] or "No Report",
        "actual_status":   actual_status or "---",
        "passed":          passed,
        "intent_detected": intent or "---",
        "trace":           turns_trace,
        "final_report":    final_report or "",
        "error":           error_msg or "",
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown("""
<div style="padding: 24px 0 8px 0;">
    <div style="font-size:1.8rem; font-weight:700; color:#e2e8f0;">📊 Evaluation Dashboard</div>
    <div style="font-size:0.9rem; color:#64748b; margin-top:4px;">
        Automated test suite — 20 scripted conversations evaluated against expected outcomes.
        The database is reset before and after the full run for isolation.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Results storage in session state ──────────────────────────────────────────
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []
if "eval_running" not in st.session_state:
    st.session_state.eval_running = False

# ── Header controls ──────────────────────────────────────────────────────────
col_btn, col_status = st.columns([1, 3])
with col_btn:
    run_btn = st.button(
        "▶  Run All 20 Tests",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.eval_running,
    )

with col_status:
    if st.session_state.eval_results:
        total  = len(st.session_state.eval_results)
        passed = sum(1 for r in st.session_state.eval_results if r["passed"])
        rate   = (passed / total * 100) if total else 0
        st.markdown(
            f"<div style='padding:10px 0; color:#94a3b8; font-size:0.9rem;'>"
            f"Last run: <b style='color:#e2e8f0'>{total}</b> tests &nbsp;|&nbsp; "
            f"<b style='color:#22c55e'>{passed} passed</b> &nbsp;|&nbsp; "
            f"<b style='color:#ef4444'>{total - passed} failed</b> &nbsp;|&nbsp; "
            f"<b style='color:#f59e0b'>{rate:.1f}% pass rate</b></div>",
            unsafe_allow_html=True,
        )

# ── Run the suite ─────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.eval_running = True
    st.session_state.eval_results = []

    # Reseed before starting
    with st.spinner("Resetting database to defaults before evaluation…"):
        initialize_database()
        seed_database()

    progress_bar = st.progress(0, text="Starting evaluation…")
    results_placeholder = st.empty()
    live_results = []

    for i, test in enumerate(TEST_SUITE):
        progress_bar.progress(
            (i) / len(TEST_SUITE),
            text=f"Running test {test['id']}/20: {test['name']}…",
        )

        result = run_single_test(test)
        live_results.append(result)
        st.session_state.eval_results = list(live_results)

        # Show a live mini-status line
        icon = "✅" if result["passed"] else "❌"
        results_placeholder.markdown(
            f"**{icon} [{test['id']}/20]** {test['name']} → "
            f"`{result['actual_status']}`"
        )

    progress_bar.progress(1.0, text="All tests complete!")

    # Reseed after finishing to restore DB to defaults
    with st.spinner("Restoring database to defaults after evaluation…"):
        initialize_database()
        seed_database()

    results_placeholder.empty()
    st.session_state.eval_running = False
    st.rerun()

# ── Summary metrics ───────────────────────────────────────────────────────────
if st.session_state.eval_results:
    results = st.session_state.eval_results
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    rate    = (passed / total * 100) if total else 0

    st.markdown("### Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number total-color">{total}</div>
            <div class="metric-label">Total Tests</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number pass-color">{passed}</div>
            <div class="metric-label">Passed</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number fail-color">{failed}</div>
            <div class="metric-label">Failed</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number rate-color">{rate:.1f}%</div>
            <div class="metric-label">Pass Rate</div>
        </div>""", unsafe_allow_html=True)

    # ── Coverage by category ──────────────────────────────────────────────────
    st.markdown("### Coverage by Category")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    cat_cols = st.columns(len(categories))
    for col, (cat, data) in zip(cat_cols, categories.items()):
        cat_rate = data["passed"] / data["total"] * 100
        color = "#22c55e" if cat_rate == 100 else "#f59e0b" if cat_rate >= 50 else "#ef4444"
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:14px;">
                <div style="font-size:1.4rem; font-weight:700; color:{color};">{data['passed']}/{data['total']}</div>
                <div class="metric-label">{cat.replace('_', ' ').title()}</div>
            </div>""", unsafe_allow_html=True)

    # ── Detailed results table ────────────────────────────────────────────────
    st.markdown("### Detailed Results")

    # Export CSV button
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "id", "name", "category", "turns_count",
        "expected_status", "actual_status", "passed", "intent_detected", "error"
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r[k] for k in writer.fieldnames})
    st.download_button(
        label="⬇  Export Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="helpbot_eval_results.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    for r in results:
        icon      = "✅" if r["passed"] else "❌"
        badge     = f'<span class="badge-pass">PASS</span>' if r["passed"] else '<span class="badge-fail">FAIL</span>'
        cat_badge = f'<span class="badge-cat">{r["category"].replace("_"," ").upper()}</span>'

        with st.expander(f"{icon}  #{r['id']} — {r['name']}", expanded=not r["passed"]):
            # Header row
            st.markdown(
                f"{badge} &nbsp; {cat_badge} &nbsp;"
                f"<span style='color:#94a3b8; font-size:0.8rem;'>"
                f"Expected: <b>{r['expected_status']}</b> &nbsp;|&nbsp; "
                f"Got: <b>{r['actual_status']}</b> &nbsp;|&nbsp; "
                f"Intent: <b>{r['intent_detected']}</b> &nbsp;|&nbsp; "
                f"Turns: <b>{r['turns_count']}</b>"
                f"</span>",
                unsafe_allow_html=True,
            )

            # Description
            st.caption(r["description"])

            if r["error"]:
                st.error(f"**Runtime error:** {r['error']}")

            # Conversation trace
            st.markdown("**Conversation Trace:**")
            for turn in r["trace"]:
                st.markdown(
                    f'<div class="trace-user"><div class="trace-label" style="color:#60a5fa;">👤 User</div>{turn["user"]}</div>',
                    unsafe_allow_html=True,
                )
                agent_text = turn["agent"][:600] + ("…" if len(turn["agent"]) > 600 else "")
                st.markdown(
                    f'<div class="trace-agent"><div class="trace-label" style="color:#4ade80;">🛠️ Agent</div>{agent_text}</div>',
                    unsafe_allow_html=True,
                )

            # Final report preview
            if r["final_report"]:
                with st.container():
                    st.markdown("**Final Report (excerpt):**")
                    st.markdown(r["final_report"][:800] + ("…" if len(r["final_report"]) > 800 else ""))

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#475569;">
        <div style="font-size:3rem;">🧪</div>
        <div style="font-size:1.1rem; font-weight:600; color:#64748b; margin-top:12px;">No results yet</div>
        <div style="font-size:0.85rem; margin-top:6px;">Click <b>Run All 20 Tests</b> above to start the evaluation suite.</div>
    </div>
    """, unsafe_allow_html=True)
