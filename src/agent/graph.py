"""
graph.py — LangGraph State Machine for the IT Help-Desk Diagnostic Agent

Flow diagram:
    START
      └─▶ orchestrate_node ──────────────────────────────────────────────────┐
            │                                                                  │
            ├─(intent=unknown)──▶ chitchat_node ──▶ END                       │
            │                                                                  │
            ├─(missing_fields)──▶ gather_info_node ──▶ END                    │
            │                                                                  │
            ├─(pending_confirmation=True)──▶ handle_confirmation_node         │
            │                                     │                           │
            │                                     ├─(confirmed)──▶ execute_action_node ─▶ generate_report_node ─▶ END
            │                                     └─(denied)────▶ generate_report_node ─▶ END
            │
            └─(all info gathered)──▶ run_diagnosis_node
                                          │
                                          ├─(sensitive action needed)──▶ ask_confirmation_node ──▶ END
                                          └─(no action / guidance only)──▶ generate_report_node ──▶ END
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.state import AgentState
from agent.gemini_client import get_llm, get_system_prompt, llm_invoke_with_retry
from tools.helpdesk_tools import (
    match_intent_from_symptoms,
    lookup_user,
    query_knowledge_base,
    analyze_diagnostic_input,
    execute_helpdesk_action,
    append_ticket_step,
    generate_resolution_report,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MAX_ITERATIONS = 5

# Maps KB article topic strings to short intent identifiers
TOPIC_TO_INTENT = {
    "VPN Connection Failure": "vpn",
    "Slow Internet or Wi-Fi Disconnects": "slow_wifi",
    "Active Directory Account Lockout": "account_lockout",
    "MFA and Password Setup": "mfa_reset",
    "Operating System / Blue Screen Crashes": "bsod",
    "Software Update Failures": "update_error",
}

# Required diagnostic fields per intent
REQUIRED_FIELDS = {
    "vpn":              ["username", "ping_result"],
    "slow_wifi":        ["ping_result"],
    "account_lockout":  ["username"],
    "mfa_reset":        ["username"],
    "bsod":             ["username", "error_code"],
    "update_error":     ["username", "disk_space", "error_code"],
    "unknown":          [],
}

# Human-readable prompts the agent uses when asking for each field
FIELD_PROMPTS = {
    "username": (
        "Could you please provide your employee **username**?\n"
        "_(e.g., `alice_smith`)_"
    ),
    "ping_result": (
        "To check your network connection, please open a terminal and run the command below, "
        "then **paste the full output** back here:\n\n"
        "- **Windows:** `ping 8.8.8.8 -n 4`\n"
        "- **macOS / Linux:** `ping -c 4 8.8.8.8`"
    ),
    "error_code": (
        "Please share the **exact error code or stop code** you are seeing.\n\n"
        "Examples:\n"
        "- Windows Update: `0x80244007`\n"
        "- Blue Screen: `PAGE_FAULT_IN_NONPAGED_AREA`"
    ),
    "disk_space": (
        "Please check how much **free disk space** you have on your system drive and report it here.\n\n"
        "- **Windows:** Open File Explorer → This PC → check C: drive\n"
        "- **macOS:** Apple Menu → About This Mac → Storage"
    ),
}

# Simple keyword sets for confirmation detection
CONFIRM_WORDS = {"yes", "ok", "sure", "confirm", "proceed", "go", "yep", "yeah", "y", "please", "do"}
DENY_WORDS    = {"no", "cancel", "abort", "stop", "nope", "n", "deny", "reject", "dont", "don't"}


# ─────────────────────────────────────────────
# Private Helpers
# ─────────────────────────────────────────────

def _classify_and_extract(message: str) -> dict:
    """
    Single LLM call that both classifies intent AND extracts username.
    Replaces the two separate calls to _classify_intent_with_gemini and
    _extract_username, halving API usage per orchestrate turn.

    Returns a dict with keys:
      'intent'   - one of: vpn | slow_wifi | account_lockout | mfa_reset |
                           bsod | update_error | unknown
      'username' - extracted username string, or None
    """
    prompt = (
        "Analyze this IT helpdesk message and return two pieces of information.\n\n"
        "1. INTENT - classify into exactly one category:\n"
        "   vpn            = VPN connection problems, cannot connect to VPN\n"
        "   slow_wifi      = slow internet, Wi-Fi drops, network speed issues\n"
        "   account_lockout = account locked, cannot log in, too many failed attempts\n"
        "   mfa_reset      = MFA, two-factor auth, authenticator app, password setup\n"
        "   bsod           = blue screen, system crash, stop code, kernel error\n"
        "   update_error   = Windows update failed, patch failure\n"
        "   unknown        = none of the above IT categories\n\n"
        "2. USERNAME - extract the employee username if present\n"
        "   Usernames are lowercase letters+digits+underscores (e.g. alice_smith)\n"
        "   If no username present, write: none\n\n"
        f"Message: \"{message}\"\n\n"
        "Reply in EXACTLY this format (two lines, no extra text):\n"
        "INTENT: <category>\n"
        "USERNAME: <username or none>"
    )
    raw = llm_invoke_with_retry([HumanMessage(content=prompt)]).strip()

    result = {"intent": "unknown", "username": None}
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            val = line.split(":", 1)[1].strip().lower()
            valid = {"vpn", "slow_wifi", "account_lockout", "mfa_reset",
                     "bsod", "update_error", "unknown"}
            result["intent"] = val if val in valid else "unknown"
        elif line.upper().startswith("USERNAME:"):
            val = line.split(":", 1)[1].strip().lower()
            if val and val != "none" and re.match(r"^[a-z0-9_]+$", val) and len(val) <= 60:
                result["username"] = val
    return result


def _extract_error_code(message: str) -> str | None:
    """Asks Gemini to extract a Windows error or BSOD stop code from user text."""
    prompt = (
        "Extract the IT error code or BSOD stop code from the message below.\n"
        "Examples: 0x80244007, PAGE_FAULT_IN_NONPAGED_AREA, CRITICAL_PROCESS_DIED.\n"
        "Reply with ONLY the code. If none is present, reply with exactly: none\n\n"
        f"Message: \"{message}\""
    )
    raw = llm_invoke_with_retry([HumanMessage(content=prompt)]).strip()
    return None if raw.lower() == "none" else raw


def _check_confirmation(message: str) -> bool | None:
    """
    Keyword-based confirmation detection.
    Returns True (confirmed), False (denied), or None (ambiguous).
    """
    tokens = set(re.findall(r"\b\w+\b", message.lower()))
    if tokens & CONFIRM_WORDS:
        return True
    if tokens & DENY_WORDS:
        return False
    return None


def _compute_missing_fields(intent: str, username: str | None, gathered: dict) -> list:
    """Returns the list of required fields that haven't been collected yet."""
    missing = []
    for field in REQUIRED_FIELDS.get(intent, []):
        if field == "username" and not username:
            missing.append("username")
        elif field != "username" and field not in gathered:
            missing.append(field)
    return missing


# ─────────────────────────────────────────────
# 1. ORCHESTRATE NODE  (entry point every turn)
# ─────────────────────────────────────────────
def orchestrate_node(state: AgentState) -> dict:
    """
    Runs at the start of every turn. Responsibilities:
      - Classify intent (if unknown) using the deterministic KB symptom matcher.
      - Extract and validate username (if missing) using Gemini.
      - Try to fill in any missing gathered fields from the user's latest message.
      - Recompute the missing_fields list.
    Routing is handled by the conditional edge function route_from_orchestrate().
    """
    # Guard: if waiting for confirmation, do nothing -- let route handle it
    if state.get("pending_confirmation"):
        return {}

    updates: dict = {}
    last_message = state["messages"][-1].content if state.get("messages") else ""

    # Guard: if a report was already generated, this is a NEW issue in the same conversation thread.
    # We must reset the diagnostic state variables so they don't leak into the new issue.
    # Note: We keep username and device_id so the user doesn't have to re-authenticate.
    if state.get("final_report"):
        reset_fields = {
            "intent": None,
            "kb_topic": None,
            "kb_id": None,
            "gathered_info": {},
            "missing_fields": [],
            "pending_confirmation": False,
            "pending_action": None,
            "ticket_id": None,
            "latest_tool_result": {},
            "steps_taken": [],
            "iterations": 0,
            "final_report": None,
            "guidance_presented": False,
            "user_confirmed_resolved": None,
        }
        updates.update(reset_fields)
        # Update local state dict so the rest of this node sees the cleared fields
        state.update(reset_fields)

    # Guard: if we presented guidance and are waiting for user's resolution answer
    if state.get("guidance_presented"):
        # Find the last AI message to give the LLM the question context
        last_agent_msg = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "ai":
                last_agent_msg = msg.content[:400]  # truncate long guidance messages
                break

        parse_prompt = (
            f"You are classifying a user's reply in an IT help-desk conversation.\n\n"
            f"The agent just asked the user (paraphrased): "
            f"\"Have the troubleshooting steps resolved your issue?\"\n"
            f"Agent's last message (excerpt): \"{last_agent_msg}\"\n\n"
            f"User's reply: \"{last_message}\"\n\n"
            f"Rules:\n"
            f"- A simple 'yes', 'yep', 'it works', 'fixed', 'resolved', 'all good', "
            f"'working now' means RESOLVED.\n"
            f"- 'no', 'still broken', 'same issue', 'didn't work', 'not fixed' means UNRESOLVED.\n"
            f"- Only say UNRESOLVED if the user clearly indicates the problem persists.\n\n"
            f"Reply with ONLY one word: 'resolved' or 'unresolved'"
        )
        answer = llm_invoke_with_retry([HumanMessage(content=parse_prompt)]).strip().lower()

        # Default to resolved for clear affirmative single-word answers (safety net)
        tokens = set(answer.split())
        if answer == "resolved" or ("resolved" in tokens and "unresolved" not in tokens):
            is_resolved = True
        elif "unresolved" in tokens:
            is_resolved = False
        else:
            # LLM returned something unexpected — fall back to keyword check on user message
            yes_words = {"yes", "yep", "yeah", "fixed", "resolved", "working", "works",
                         "solved", "great", "good", "done", "all good", "it works"}
            no_words  = {"no", "nope", "not", "still", "broken", "same", "didn't",
                         "doesn't", "issue", "problem"}
            user_tokens = set(last_message.lower().split())
            has_yes = bool(user_tokens & yes_words)
            has_no  = bool(user_tokens & no_words)
            is_resolved = has_yes and not has_no

        return {"user_confirmed_resolved": is_resolved, "guidance_presented": False}

    # -- Step 1: Classify intent ----------------------------------------------
    if not state.get("intent") or state.get("intent") == "unknown":
        # First try the fast deterministic KB matcher (no API call)
        match = match_intent_from_symptoms(last_message)
        topic = match.get("topic")
        if topic and topic in TOPIC_TO_INTENT:
            updates["intent"]   = TOPIC_TO_INTENT[topic]
            updates["kb_topic"] = topic
            updates["kb_id"]    = match.get("kb_id")
        else:
            # KB matcher failed, use Gemini to classify and optionally extract username
            extracted = _classify_and_extract(last_message)
            if extracted["intent"] != "unknown":
                updates["intent"] = extracted["intent"]
                # Try to find a matching KB article
                category_map = {
                    "vpn": "network", "slow_wifi": "network",
                    "account_lockout": "account", "mfa_reset": "account",
                    "bsod": "application", "update_error": "application",
                }
                kb_cat = category_map.get(extracted["intent"])
                if kb_cat:
                    kb_rows = query_knowledge_base(category=kb_cat)
                    intent_to_topic = {v: k for k, v in TOPIC_TO_INTENT.items()}
                    target_topic = intent_to_topic.get(extracted["intent"])
                    for row in kb_rows:
                        if row.get("topic") == target_topic:
                            updates["kb_topic"] = row["topic"]
                            updates["kb_id"]    = row["id"]
                            break
            else:
                updates["intent"] = "unknown"

            if extracted.get("username"):
                updates["extracted_username_candidate"] = extracted["username"]

    current_intent = updates.get("intent", state.get("intent", "unknown"))

    # -- Step 2: Extract username ---------------------------------------------
    if not state.get("username") and not updates.get("username"):
        # Did we extract it in the previous step?
        candidate = updates.pop("extracted_username_candidate", None)
        if not candidate:
            # Run the extractor call
            extracted = _classify_and_extract(last_message)
            candidate = extracted.get("username")
        
        if candidate:
            user_data = lookup_user(candidate)
            if user_data.get("found"):
                updates["username"]  = candidate
                updates["device_id"] = user_data.get("device_id")
                gathered = dict(state.get("gathered_info") or {})
                gathered["user_data"] = user_data
                updates["gathered_info"] = gathered
            else:
                # Bug 2 fix: username was provided but not found in DB.
                # Set a flag so gather_info_node can inform the user.
                updates["username_not_found"] = candidate

    # -- Step 3: Fill in gathered fields from user message --------------------
    current_username = updates.get("username", state.get("username"))
    current_gathered = dict(state.get("gathered_info") or {})
    previous_missing = state.get("missing_fields") or []

    if previous_missing:
        # Only populate the field that the user was actually prompted for
        prompted_field = previous_missing[0]
        if prompted_field != "username" and prompted_field not in current_gathered:
            if prompted_field in ("ping_result", "disk_space"):
                current_gathered[prompted_field] = last_message
            elif prompted_field == "error_code":
                code = _extract_error_code(last_message)
                if code:
                    current_gathered[prompted_field] = code
    else:
        # Bug 1 fix: First turn — no previous prompts yet.
        # Opportunistically extract error_code from the initial problem description
        # so it isn't lost when the user mentions it upfront.
        if current_intent == "bsod" and "error_code" not in current_gathered:
            code = _extract_error_code(last_message)
            if code:
                current_gathered["error_code"] = code
        elif current_intent == "update_error" and "error_code" not in current_gathered:
            code = _extract_error_code(last_message)
            if code:
                current_gathered["error_code"] = code

    updates["gathered_info"]  = current_gathered

    # -- Step 4: Recompute missing fields -------------------------------------
    updates["missing_fields"] = _compute_missing_fields(
        current_intent, current_username, current_gathered
    )

    return updates


# ─────────────────────────────────────────────
# 2. CHITCHAT / CLARIFICATION NODE
# ─────────────────────────────────────────────
def chitchat_node(state: AgentState) -> dict:
    """
    Handles greetings and messages where intent could not be classified.
    Calls Gemini with the full conversation history so it can respond naturally
    and ask the user to describe their IT problem.
    """
    messages_for_llm = [SystemMessage(content=get_system_prompt())] + list(state["messages"])
    response = llm_invoke_with_retry(messages_for_llm)
    return {"messages": [AIMessage(content=response)]}


# ─────────────────────────────────────────────
# 3. GATHER INFO NODE
# ─────────────────────────────────────────────
def gather_info_node(state: AgentState) -> dict:
    """
    Asks the user for the next missing piece of diagnostic information.
    Uses Gemini to wrap the structured field prompt in a conversational tone,
    acknowledging what the user just said before asking the next question.
    Returns an AI message and ends the turn (waits for user reply).
    """
    missing = state.get("missing_fields") or []
    if not missing:
        return {}

    next_field = missing[0]
    field_hint = FIELD_PROMPTS.get(next_field, f"Please provide your {next_field}.")
    last_user  = state["messages"][-1].content if state.get("messages") else ""

    # Bug 2 fix: If a username lookup just failed, prepend a warning
    username_warning = ""
    bad_username = state.get("username_not_found")
    if bad_username and next_field == "username":
        username_warning = (
            f"IMPORTANT: The user just provided the username '{bad_username}' but it was "
            f"NOT FOUND in the employee directory. You MUST inform them that the username "
            f"was not recognized and ask them to double-check and provide a valid one.\n\n"
        )

    # First time gathering for this intent -- fetch KB diagnostic steps to reference
    kb_context = ""
    if state.get("kb_topic") and not state.get("steps_taken"):
        kb_results = query_knowledge_base(topic=state["kb_topic"])
        if kb_results:
            steps = kb_results[0].get("diagnostic_steps", [])
            kb_context = "\nDiagnostic protocol from Knowledge Base:\n" + "\n".join(f"  - {s}" for s in steps)

    context_prompt = (
        f"{username_warning}"
        f"The user said: \"{last_user}\"\n"
        f"Intent classified: {state.get('intent', 'unknown')}\n"
        f"{kb_context}\n\n"
        f"You need to ask the user for the following information next:\n"
        f"  Field: {next_field}\n"
        f"  Suggested prompt: {field_hint}\n\n"
        f"Write ONE concise, professional response. Acknowledge what the user said (if meaningful), "
        f"then clearly ask for the required information."
    )
    response = llm_invoke_with_retry([
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=context_prompt),
    ])
    return {"messages": [AIMessage(content=response)], "username_not_found": None}


# ─────────────────────────────────────────────
# 4. RUN DIAGNOSIS NODE
# ─────────────────────────────────────────────
def run_diagnosis_node(state: AgentState) -> dict:
    """
    Runs the deterministic Analysis Tool based on the classified intent and
    gathered diagnostic data. Sets pending_confirmation=True for sensitive
    actions (e.g. account unlocks). Increments the iteration counter.
    """
    intent      = state.get("intent", "unknown")
    username    = state.get("username")
    gathered    = state.get("gathered_info") or {}
    steps       = list(state.get("steps_taken") or [])
    iterations  = (state.get("iterations") or 0) + 1

    tool_result: dict = {}

    if intent in ("account_lockout", "mfa_reset"):
        tool_result = analyze_diagnostic_input("account_check", "", username=username)

    elif intent in ("vpn", "slow_wifi"):
        tool_result = analyze_diagnostic_input("ping", gathered.get("ping_result", ""))

    elif intent == "bsod":
        tool_result = analyze_diagnostic_input("error_code", gathered.get("error_code", ""))

    elif intent == "update_error":
        disk_res = analyze_diagnostic_input("disk_space", gathered.get("disk_space", ""))
        err_res  = analyze_diagnostic_input("error_code", gathered.get("error_code", ""))
        # Use the more severe result
        tool_result = disk_res if disk_res["status"] == "error" else err_res

    # Log this diagnostic step
    step_desc = (
        f"[{intent.upper()}] Analysis → {tool_result.get('detected_issue', 'N/A')} "
        f"(status: {tool_result.get('status', '?')}, confidence: {tool_result.get('confidence_score', 0.0)})"
    )
    steps.append(step_desc)

    # Determine if a sensitive action is needed
    pending_action        = None
    pending_confirmation  = False
    details = tool_result.get("details", {})

    if intent == "account_lockout" and details.get("is_locked"):
        pending_action       = "unlock_account"
        pending_confirmation = True

    return {
        "latest_tool_result":  tool_result,
        "iterations":          iterations,
        "steps_taken":         steps,
        "pending_action":      pending_action,
        "pending_confirmation": pending_confirmation,
    }

# ─────────────────────────────────────────────
# 4b. PRESENT GUIDANCE NODE (new)
# ─────────────────────────────────────────────
def present_guidance_node(state: AgentState) -> dict:
    """
    After initial diagnosis (or after a sensitive action is executed), presents
    KB resolution steps to the user and asks if the issue is now resolved.
    The conversation does NOT end until the user confirms resolution or failure.
    """
    intent = state.get("intent", "unknown")
    tool_result = state.get("latest_tool_result") or {}
    kb_topic = state.get("kb_topic")
    detected_issue = tool_result.get("detected_issue", "An issue was detected.")
    recommended_fix = tool_result.get("recommended_fix", "")

    # After a successful action (e.g. account unlock), the tool result carries a message
    action_message = tool_result.get("message", "")

    # Fetch KB *resolution* steps (not diagnostic steps — diagnosis already ran)
    kb_resolution_steps = []
    kb_diagnostic_steps = []
    if kb_topic:
        kb_results = query_knowledge_base(topic=kb_topic)
        if kb_results:
            kb_resolution_steps = kb_results[0].get("resolution_steps", [])
            kb_diagnostic_steps  = kb_results[0].get("diagnostic_steps", [])

    # Prefer resolution steps; fall back to diagnostic steps if none
    kb_steps = kb_resolution_steps if kb_resolution_steps else kb_diagnostic_steps

    steps_text = ""
    if kb_steps:
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(kb_steps))

    context_prompt = (
        f"You are an IT help-desk agent. You just finished diagnostics for a '{intent}' issue.\n"
        f"Diagnostic finding: {detected_issue}\n"
    )
    if action_message:
        context_prompt += f"Action taken: {action_message}\n"
    if recommended_fix:
        context_prompt += f"Recommended fix: {recommended_fix}\n"
    context_prompt += "\n"

    if steps_text:
        context_prompt += (
            f"The Knowledge Base provides these resolution steps for the user:\n{steps_text}\n\n"
        )

    context_prompt += (
        f"Write a professional, friendly response that:\n"
        f"1. Briefly summarises what the diagnosis found (or what action was just taken)\n"
        f"2. Lists the resolution steps the user should try NOW\n"
        f"3. Ends by asking: 'Has this resolved your issue?' so you know whether to close or escalate\n\n"
        f"Be concise. Use markdown formatting."
    )

    response = llm_invoke_with_retry([
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=context_prompt),
    ])

    steps = list(state.get("steps_taken") or [])
    steps.append(f"[GUIDANCE] Presented resolution steps to user for: {intent}")

    return {
        "messages": [AIMessage(content=response)],
        "guidance_presented": True,
        "steps_taken": steps,
    }


# ─────────────────────────────────────────────
# 5. ASK CONFIRMATION NODE
# ─────────────────────────────────────────────
def ask_confirmation_node(state: AgentState) -> dict:
    """
    Generates a structured confirmation request message to the user.
    The graph ends after this node and waits for the user's yes/no response
    on the next turn (where orchestrate routes to handle_confirmation_node).
    """
    action      = state.get("pending_action")
    username    = state.get("username", "the user")
    tool_result = state.get("latest_tool_result") or {}
    finding     = tool_result.get("detected_issue", "An issue was detected.")

    messages_map = {
        "unlock_account": (
            "[ACCOUNT LOCKOUT DETECTED]\n\n"
            f"Finding: {finding}\n\n"
            f"I can unlock the Active Directory account for '{username}' right now.\n\n"
            "SECURITY NOTICE: I need your explicit confirmation before making any account changes.\n\n"
            "Please type 'Yes' to confirm the unlock, or 'No' to cancel."
        ),
    }

    msg = messages_map.get(
        action,
        f"I need your confirmation to proceed with: {action}. "
        f"Type 'Yes' to confirm or 'No' to cancel."
    )
    return {"messages": [AIMessage(content=msg)]}


# ─────────────────────────────────────────────
# 6. HANDLE CONFIRMATION NODE
# ─────────────────────────────────────────────
def handle_confirmation_node(state: AgentState) -> dict:
    """
    Processes the user's yes/no response to a sensitive action confirmation request.
    - Confirmed → clears pending_confirmation, keeps pending_action (execute_action will run).
    - Denied    → clears both, generates a cancellation message.
    """
    last_message = state["messages"][-1].content if state.get("messages") else ""

    # Try keyword-based detection first (fast path)
    confirmed = _check_confirmation(last_message)

    if confirmed is None:
        # Ambiguous reply — ask Gemini to interpret
        parse_prompt = (
            f"The user was asked to confirm or deny an IT account action.\n"
            f"Their reply was: \"{last_message}\"\n"
            f"Did they confirm (yes) or deny (no) the action? Reply with ONLY: yes or no"
        )
        answer = llm_invoke_with_retry([HumanMessage(content=parse_prompt)]).strip().lower()
        confirmed = answer == "yes"

    if confirmed:
        # Keep pending_action so route_from_confirmation sends us to execute_action
        return {"pending_confirmation": False}
    else:
        return {
            "pending_confirmation": False,
            "pending_action":       None,
            "messages": [AIMessage(content=(
                "Understood. The account unlock has been **cancelled**.\n\n"
                "I will generate a triage summary of the findings. "
                "If you change your mind or need further help, please start a new request."
            ))],
        }


# ─────────────────────────────────────────────
# 7. EXECUTE ACTION NODE
# ─────────────────────────────────────────────
def execute_action_node(state: AgentState) -> dict:
    """
    Executes the confirmed sensitive action against the SQLite database.
    Appends the action result to the steps log and to the active ticket (if any).
    """
    action    = state.get("pending_action")
    username  = state.get("username")
    ticket_id = state.get("ticket_id")
    steps     = list(state.get("steps_taken") or [])

    if not action or not username:
        return {}

    result = execute_helpdesk_action(
        action_type=action,
        username=username,
        details={"ticket_id": ticket_id},
        confirmed=True,
    )

    step_msg = (
        f"[ACTION] {action} for '{username}' → "
        f"{result.get('status', 'unknown')}: {result.get('message', '')}"
    )
    steps.append(step_msg)

    if ticket_id:
        append_ticket_step(ticket_id, step_msg)

    return {
        "latest_tool_result": result,
        "pending_action":     None,
        "steps_taken":        steps,
        "messages": [AIMessage(
            content=f"Action completed: {result.get('message', 'Done.')}"
        )],
    }


# ─────────────────────────────────────────────
# 8. GENERATE REPORT NODE
# ─────────────────────────────────────────────
def generate_report_node(state: AgentState) -> dict:
    """
    Final node: creates the database ticket (if not yet done), closes it as
    Resolved or Escalated based on diagnostic outcomes and iteration count,
    then generates and returns the markdown triage report.
    """
    username    = state.get("username")   # may be None — create_ticket handles FK safely
    device_id   = state.get("device_id")
    intent      = state.get("intent") or "general"
    steps       = list(state.get("steps_taken") or [])
    tool_result = state.get("latest_tool_result") or {}
    iterations  = state.get("iterations") or 0
    ticket_id   = state.get("ticket_id")

    symptoms_summary = (
        f"[{intent.upper()}] "
        + tool_result.get("detected_issue", "Issue reported via Help-Desk Agent.")
    )

    # ── Create ticket (lazy — only if not already open) ──────────────────────
    if not ticket_id:
        ticket_res = execute_helpdesk_action("create_ticket", username, details={
            "device_id":  device_id,
            "category":   intent,
            "symptoms":   symptoms_summary,
            "steps_taken": steps,
        })
        ticket_id = ticket_res.get("ticket_id")
    else:
        # Append any new steps accumulated this session
        for step in steps:
            append_ticket_step(ticket_id, step)

    # ── Decide resolution status ──────────────────────────────────────────────
    action_was_taken  = any("[ACTION]" in s for s in steps)
    diagnostic_error  = tool_result.get("status") == "error"
    needs_specialist  = intent in ("vpn", "bsod") and diagnostic_error and not action_was_taken
    max_iter_exceeded = iterations >= MAX_ITERATIONS

    should_escalate = needs_specialist or max_iter_exceeded
    user_resolved   = state.get("user_confirmed_resolved")

    if user_resolved is True:
        should_escalate = False
    elif user_resolved is False:
        should_escalate = True

    if should_escalate:
        reason = (
            f"Issue unresolved after {iterations} diagnostic iteration(s). "
            f"Requires Tier-2 specialist for: {intent.upper()}."
        )
        if user_resolved is False:
            reason = f"User confirmed issue is still present after troubleshooting. Requires Tier-2 specialist for: {intent.upper()}."
            
        execute_helpdesk_action("escalate_ticket", username, details={
            "ticket_id": ticket_id,
            "reason":    reason,
        })
    else:
        execute_helpdesk_action("resolve_ticket", username, details={"ticket_id": ticket_id})

    # ── Generate the final markdown report ───────────────────────────────────
    report = generate_resolution_report(ticket_id)

    return {
        "ticket_id":   ticket_id,
        "final_report": report,
        "messages":    [AIMessage(content=report)],
    }


# ─────────────────────────────────────────────
# ROUTING FUNCTIONS (pure — read state, return node name)
# ─────────────────────────────────────────────

def route_from_orchestrate(state: AgentState) -> str:
    """After orchestrate: decide which node handles this turn."""
    if state.get("pending_confirmation"):
        return "handle_confirmation"

    # Post-guidance: user has responded whether their issue is resolved
    if state.get("user_confirmed_resolved") is not None:
        return "generate_report"

    intent = state.get("intent", "unknown")
    if not intent or intent == "unknown":
        return "chitchat"
    if state.get("missing_fields"):
        return "gather_info"
    if (state.get("iterations") or 0) >= MAX_ITERATIONS:
        return "generate_report"
    return "run_diagnosis"


def route_from_diagnosis(state: AgentState) -> str:
    """After run_diagnosis: go to ask confirmation, present guidance, or generate report."""
    if state.get("pending_confirmation"):
        return "ask_confirmation"
    # If guidance was already shown this session, go straight to report
    # (avoids presenting the same tips twice in a loop)
    if state.get("guidance_presented"):
        return "generate_report"
    return "present_guidance"


def route_from_confirmation(state: AgentState) -> str:
    """After handle_confirmation: execute action (confirmed) or generate report (denied)."""
    if state.get("pending_action"):
        return "execute_action"
    return "generate_report"


# ─────────────────────────────────────────────
# BUILD & COMPILE THE GRAPH
# ─────────────────────────────────────────────

def build_graph():
    """
    Constructs, wires, and compiles the LangGraph state machine.
    Uses MemorySaver as the checkpointer so state persists across
    conversation turns (identified by thread_id in config).

    Returns:
        CompiledGraph: The ready-to-invoke agent graph.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("orchestrate",          orchestrate_node)
    builder.add_node("chitchat",             chitchat_node)
    builder.add_node("gather_info",          gather_info_node)
    builder.add_node("run_diagnosis",        run_diagnosis_node)
    builder.add_node("ask_confirmation",     ask_confirmation_node)
    builder.add_node("handle_confirmation",  handle_confirmation_node)
    builder.add_node("execute_action",       execute_action_node)
    builder.add_node("generate_report",      generate_report_node)
    builder.add_node("present_guidance",     present_guidance_node)

    # Entry point
    builder.set_entry_point("orchestrate")

    # Conditional routing from orchestrate
    builder.add_conditional_edges(
        "orchestrate",
        route_from_orchestrate,
        {
            "chitchat":            "chitchat",
            "gather_info":         "gather_info",
            "run_diagnosis":       "run_diagnosis",
            "handle_confirmation": "handle_confirmation",
            "generate_report":     "generate_report",
        },
    )

    # Nodes that end the turn (wait for next user message)
    builder.add_edge("chitchat",         END)
    builder.add_edge("gather_info",      END)
    builder.add_edge("ask_confirmation", END)
    builder.add_edge("present_guidance", END)

    # Conditional routing from run_diagnosis
    builder.add_conditional_edges(
        "run_diagnosis",
        route_from_diagnosis,
        {
            "ask_confirmation": "ask_confirmation",
            "present_guidance": "present_guidance",
            "generate_report":  "generate_report",
        },
    )

    # Conditional routing from handle_confirmation
    builder.add_conditional_edges(
        "handle_confirmation",
        route_from_confirmation,
        {
            "execute_action":  "execute_action",
            "generate_report": "generate_report",
        },
    )

    # After executing an action (e.g. unlock), ask the user if it resolved the issue
    # rather than jumping straight to the final report
    builder.add_edge("execute_action",  "present_guidance")
    builder.add_edge("generate_report", END)

    # Compile with in-memory state persistence
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Singleton graph instance (imported by app.py and tests)
graph = build_graph()
