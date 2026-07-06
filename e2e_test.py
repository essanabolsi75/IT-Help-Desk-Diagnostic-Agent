"""
e2e_test.py — End-to-end conversational test for the IT Help-Desk Diagnostic Agent.

Simulates three real conversation flows through the LangGraph state machine:
  Scenario A: VPN failure (network -> gather info -> diagnose -> escalate report)
  Scenario B: Account lockout (account -> gather -> diagnose -> confirm -> action -> resolve report)
  Scenario C: Unknown intent / chitchat (greeting -> clarification response)

Each turn calls graph.invoke() with the user message and the same thread_id
so the MemorySaver checkpointer maintains state between turns.
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Seed a fresh database for the test
os.environ["DB_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_test.db")

from db.database import initialize_database, seed_database
from agent.graph import graph
from langchain_core.messages import HumanMessage

initialize_database()
seed_database()

DIVIDER = "=" * 70


def chat(thread_id: str, user_message: str, turn_label: str = ""):
    """Send one user message and print the agent's response."""
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
    print("  AGENT >\n")
    # Indent agent response for readability
    for line in (last_ai or "[no response]").split("\n"):
        print("    " + line.encode('ascii', errors='replace').decode('ascii'))

    # Print key state fields for inspection
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


# -----------------------------------------------------------------------------
# SCENARIO A — VPN Failure
# Multi-turn: Intent classified -> ask username -> ask ping -> diagnose -> report
# -----------------------------------------------------------------------------
print(f"\n{DIVIDER}")
print("  SCENARIO A: VPN Connection Failure")
print(DIVIDER)

thread_a = f"vpn-test-{uuid.uuid4().hex[:8]}"

chat(thread_a, "Hi, my VPN keeps disconnecting. I can't reach the office network at all.", "Turn 1 — Describe problem")
chat(thread_a, "My username is alice_smith", "Turn 2 — Provide username")
chat(thread_a,
     "Pinging 8.8.8.8 with 32 bytes of data:\n"
     "Request timed out.\nRequest timed out.\nRequest timed out.\nRequest timed out.\n"
     "Ping statistics for 8.8.8.8:\n"
     "    Packets: Sent = 4, Received = 0, Lost = 4 (100% loss)",
     "Turn 3 — Paste ping output (100% loss)")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO B — Account Lockout with Confirmation Gate
# Multi-turn: Intent → ask username → diagnose (locked) → confirm gate → unlock → resolve
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n\n{DIVIDER}")
print("  SCENARIO B: Account Lockout + Confirmation Gate")
print(DIVIDER)

thread_b = f"lockout-test-{uuid.uuid4().hex[:8]}"

chat(thread_b, "I can't log in. My account seems to be locked.", "Turn 1 — Describe problem")
chat(thread_b, "My username is bob_jones", "Turn 2 — Provide username")
chat(thread_b, "Yes, please go ahead and unlock it", "Turn 3 — User confirms unlock")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO C — Chitchat / Unknown Intent
# Single turn: greeting with no IT issue → agent asks what's wrong
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n\n{DIVIDER}")
print("  SCENARIO C: Chitchat / Unknown Intent")
print(DIVIDER)

thread_c = f"chitchat-test-{uuid.uuid4().hex[:8]}"

chat(thread_c, "Hello there! How are you doing today?", "Turn 1 — Greeting only")

print(f"\n\n{DIVIDER}")
print("  ALL SCENARIOS COMPLETE")
print(DIVIDER)

# Cleanup
if os.path.exists(os.environ["DB_PATH"]):
    os.remove(os.environ["DB_PATH"])
    print("\n  Test database cleaned up.")
