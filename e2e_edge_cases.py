"""
e2e_edge_cases.py -- Additional edge-case scenarios for stress-testing the agent.

Scenario D: User provides username AND problem in the SAME first message
Scenario E: User says "No" to the confirmation gate (account unlock denied)
Scenario F: BSOD / blue screen with error code
Scenario G: User provides an unknown/invalid username
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

os.environ["DB_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_edge_test.db")

from db.database import initialize_database, seed_database
from agent.graph import graph
from langchain_core.messages import HumanMessage

initialize_database()
seed_database()

DIVIDER = "=" * 70


def chat(thread_id, user_message, turn_label=""):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    last_ai = None
    for msg in reversed(result["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            last_ai = msg.content
            break

    print("\n" + "-" * 60)
    if turn_label:
        print("  " + turn_label)
    print("-" * 60)
    print("  USER  > " + user_message[:120])
    print("-" * 60)
    print("  AGENT >")
    for line in (last_ai or "[no response]").split("\n"):
        print("    " + line.encode('ascii', errors='replace').decode('ascii'))

    state = graph.get_state(config).values
    print(
        "\n  [STATE] intent={} | username={} | missing={} | pending_confirm={} | ticket_id={} | iterations={}".format(
            state.get("intent"),
            state.get("username"),
            state.get("missing_fields"),
            state.get("pending_confirmation"),
            state.get("ticket_id"),
            state.get("iterations"),
        )
    )


# ── SCENARIO D: Username + problem in one message ──────────────────────────
print(f"\n{DIVIDER}")
print("  SCENARIO D: Username + Problem in Single Message")
print(DIVIDER)

thread_d = uuid.uuid4().hex[:8]
chat(thread_d, "Hi, I'm alice_smith and my VPN keeps dropping", "Turn 1 - Combined intro")
chat(thread_d, "Pinging 8.8.8.8 with 32 bytes of data:\nReply from 8.8.8.8: bytes=32 time=14ms TTL=118\nReply from 8.8.8.8: bytes=32 time=12ms TTL=118\nReply from 8.8.8.8: bytes=32 time=15ms TTL=118\nReply from 8.8.8.8: bytes=32 time=11ms TTL=118\n\nPing statistics for 8.8.8.8:\n    Packets: Sent = 4, Received = 4, Lost = 0 (0% loss)", "Turn 2 - Ping OK (0% loss)")


# ── SCENARIO E: Confirmation denied ────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  SCENARIO E: Account Lockout - User Says NO to Unlock")
print(DIVIDER)

thread_e = uuid.uuid4().hex[:8]
chat(thread_e, "My account is locked out, username is bob_jones", "Turn 1 - Lockout + username")
chat(thread_e, "No, don't unlock it. I'll contact my manager first.", "Turn 2 - Deny confirmation")


# ── SCENARIO F: BSOD with error code ──────────────────────────────────────
print(f"\n{DIVIDER}")
print("  SCENARIO F: Blue Screen of Death with Stop Code")
print(DIVIDER)

thread_f = uuid.uuid4().hex[:8]
chat(thread_f, "My computer crashed with a blue screen showing PAGE_FAULT_IN_NONPAGED_AREA", "Turn 1 - BSOD report")
chat(thread_f, "My username is alice_smith", "Turn 2 - Provide username")


# ── SCENARIO G: Unknown username ──────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  SCENARIO G: User Provides Non-Existent Username")
print(DIVIDER)

thread_g = uuid.uuid4().hex[:8]
chat(thread_g, "I can't connect to VPN", "Turn 1 - VPN issue")
chat(thread_g, "My username is john_nonexistent", "Turn 2 - Invalid username")


print(f"\n{DIVIDER}")
print("  ALL EDGE-CASE SCENARIOS COMPLETE")
print(DIVIDER)

# Cleanup
try:
    os.remove(os.environ["DB_PATH"])
    print("\n  Test database cleaned up.")
except OSError:
    pass
