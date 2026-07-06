from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Full working memory and short-term memory for the helpdesk agent.

    Short-Term Memory:
        messages: Full conversation history (human + AI turns). Managed by LangGraph's
                  add_messages reducer which accumulates messages across turns.

    Working Memory (reset or updated each turn):
        username:             Validated employee username (e.g. 'alice_smith').
        device_id:            Device record ID associated with the user (e.g. 'LAPTOP-ALICE11').
        intent:               Classified troubleshooting category:
                              'vpn' | 'slow_wifi' | 'account_lockout' | 'mfa_reset' |
                              'bsod' | 'update_error' | 'unknown'.
        kb_topic:             Matched Knowledge Base article topic string.
        kb_id:                Matched Knowledge Base article row ID.
        gathered_info:        Key-value store of collected diagnostic values:
                              'ping_result', 'disk_space', 'error_code', 'user_data'.
        missing_fields:       List of field names still needed from the user before
                              diagnosis can run.
        pending_confirmation: True when the agent is waiting for the user to explicitly
                              confirm a sensitive action (e.g. unlocking an account).
        pending_action:       The action type string to execute after confirmation
                              (e.g. 'unlock_account').
        ticket_id:            Database ticket ID once created during the session.
        latest_tool_result:   The last dict returned by any tool (analysis, action, etc.).
        steps_taken:          Running log of diagnostic steps performed this session.
        iterations:           Counter of how many times run_diagnosis_node has executed.
                              Bounded at MAX_ITERATIONS (5) to prevent infinite loops.
        final_report:         The completed markdown triage report string.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    username: Optional[str]
    device_id: Optional[str]
    intent: Optional[str]
    kb_topic: Optional[str]
    kb_id: Optional[int]
    gathered_info: dict
    missing_fields: list
    pending_confirmation: bool
    pending_action: Optional[str]
    ticket_id: Optional[int]
    latest_tool_result: dict
    steps_taken: list
    iterations: int
    final_report: Optional[str]
