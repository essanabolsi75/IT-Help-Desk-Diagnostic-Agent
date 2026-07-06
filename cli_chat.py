"""
cli_chat.py -- Interactive command-line interface for the IT Help-Desk Agent.

Allows testing the agent directly from the terminal. Keeps track of the thread ID
and prints the agent's internal state after each turn so you can verify the logic.
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from db.database import initialize_database, seed_database
from agent.graph import graph
from langchain_core.messages import HumanMessage

# Initialize and seed database if not already done
if not os.path.exists("helpdesk.db"):
    initialize_database()
    seed_database()

def main():
    thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
    
    print("=" * 70)
    print("  HelpBot - Interactive IT Triage CLI Client")
    print(f"  Session Thread ID: {thread_id}")
    print("  Type 'exit' or 'quit' to end. Type 'reset' to start a new thread.")
    print("=" * 70)
    print("\nAGENT > Hello! I'm HelpBot, your IT Help-Desk assistant. How can I help you today?")
    
    while True:
        try:
            user_input = input("\nUSER  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
            
        if user_input.lower() == "reset":
            thread_id = uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": thread_id}}
            print("\n" + "=" * 70)
            print(f"  New Session Started (Thread ID: {thread_id})")
            print("=" * 70)
            print("\nAGENT > Thread reset. How can I help you?")
            continue
            
        # Send message to LangGraph agent
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            
            # Extract last AI response
            last_ai = None
            for msg in reversed(result["messages"]):
                if msg.__class__.__name__ == "AIMessage":
                    last_ai = msg.content
                    break
                    
            print("\nAGENT >")
            if last_ai:
                # Clean prints for terminal safety
                clean_text = last_ai.encode("ascii", errors="replace").decode("ascii")
                for line in clean_text.splitlines():
                    print(f"  {line}")
            else:
                print("  [No response from agent]")
                
            # Inspect the state after the turn
            state = graph.get_state(config).values
            print("\n" + "-" * 50)
            print(f"  [Agent Memory State]")
            print(f"  Intent:           {state.get('intent')}")
            print(f"  Username:         {state.get('username')}")
            print(f"  Missing Fields:   {state.get('missing_fields')}")
            print(f"  Pending Action:   {state.get('pending_action')}")
            print(f"  Pending Confirm:  {state.get('pending_confirmation')}")
            print(f"  Ticket ID:        {state.get('ticket_id')}")
            print(f"  Iteration Count:  {state.get('iterations')}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to run agent: {e}")

if __name__ == "__main__":
    main()
