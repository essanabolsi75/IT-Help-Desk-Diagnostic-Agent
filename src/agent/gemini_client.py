"""
gemini_client.py — Gemini API client using the official google-genai SDK.

Uses the new google-genai SDK which natively supports the AQ. authentication
key format introduced by Google in 2025. The langchain-google-genai wrapper
does not yet support this new key format.
"""

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Model fallback chain — tried in order until one succeeds
# ─────────────────────────────────────────────────────────────────
MODELS_FALLBACK = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]

# ─────────────────────────────────────────────────────────────────
# System prompt injected into every Gemini call
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are HelpBot, a professional IT Help-Desk Diagnostic Agent for a corporate environment.
You help employees troubleshoot three categories of IT issues:
  1. Network Connectivity (VPN failures, slow Wi-Fi, DNS issues)
  2. Account & Authentication (account lockouts, expired passwords, MFA setup)
  3. OS & Application Problems (blue screens, Windows update failures)

Behaviour rules you must always follow:
- Be concise, professional, and empathetic.
- Ask only ONE question at a time. Never overwhelm the user.
- When instructing users to run commands, format the command in a code block.
- Never guess or hallucinate. Only use information already provided to you.
- Always acknowledge what the user just told you before asking your next question.
- When you need command output from the user (like ping results), tell them EXACTLY what to run and what to paste back.
- If an issue is beyond the scope of your diagnostic tools, clearly say so and tell the user you will escalate to Tier-2 support.
- Never perform any account changes, unlocks, or system modifications without explicit user confirmation.

Grounding rule: All troubleshooting steps you suggest must come from the corporate Knowledge Base.
Do not invent steps that are not grounded in the provided diagnostic data.
"""


def get_client() -> genai.Client:
    """
    Returns an initialized google-genai Client using the GEMINI_API_KEY.
    Supports both classic AIzaSy keys and the new AQ. authentication keys.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment.\n"
            "Steps to fix:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Paste your API key from https://aistudio.google.com/app/apikey\n"
        )
    return genai.Client(api_key=api_key)


def get_system_prompt() -> str:
    """Returns the agent system prompt string."""
    return SYSTEM_PROMPT


def llm_invoke_with_retry(messages: list, max_retries: int = 4) -> str:
    """
    Calls Gemini with exponential backoff retry for 429 quota errors.
    Accepts a list of message dicts OR LangChain-style message objects
    and normalises them into the google-genai Content format.

    Args:
        messages:    List of messages. Supported formats:
                     - dict with 'role' and 'content' keys
                     - LangChain HumanMessage / AIMessage / SystemMessage objects
        max_retries: Maximum number of retry attempts on 429 (default 4).

    Returns:
        str: The model's response text.

    Raises:
        Exception: Re-raises the last error after all retries are exhausted.
    """
    client = get_client()

    # ── Normalise messages into google-genai Contents ─────────────────────────
    system_text = SYSTEM_PROMPT
    contents: list[types.Content] = []

    for msg in messages:
        # Handle LangChain message objects
        if hasattr(msg, "__class__"):
            cls = msg.__class__.__name__
            text = msg.content if hasattr(msg, "content") else str(msg)
            if cls == "SystemMessage":
                system_text = text          # Override system prompt
                continue
            elif cls == "HumanMessage":
                role = "user"
            elif cls == "AIMessage":
                role = "model"
            else:
                role = "user"
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_text = text
                continue
        else:
            continue

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=text)]
            )
        )

    # -- Retry loop with per-model fallback ---------------------------------
    # 429 RESOURCE_EXHAUSTED -> skip immediately to next model (quota is gone)
    # 503 UNAVAILABLE        -> retry same model with short backoff
    # Anything else          -> re-raise immediately
    last_error = None
    for model in MODELS_FALLBACK:
        model_failed = False
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_text,
                        temperature=0.3,
                        max_output_tokens=2048,
                    ),
                )
                return response.text
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Quota is exhausted -- no point retrying this model
                    last_error = e
                    print(f"  [FALLBACK] {model} quota exhausted -> trying next model...")
                    model_failed = True
                    break
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    # Transient overload -- retry with backoff on same model
                    last_error = e
                    if attempt < max_retries:
                        wait = 2 ** attempt * 3   # 3s -> 6s -> 12s -> 24s
                        print(f"  [503 RETRY] {model} overloaded. Retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait)
                    else:
                        print(f"  [FALLBACK] {model} still unavailable -> trying next model...")
                        model_failed = True
                        break
                else:
                    raise  # Unknown error -- surface immediately
        if not model_failed:
            break  # Should not reach here if return happened, but safety guard
    # All models exhausted -- return a graceful fallback instead of crashing
    print("  [QUOTA EXHAUSTED] All models quota-limited. Running in degraded mode.")
    return "[QUOTA EXHAUSTED: Free tier daily limit reached. The agent is running in limited mode. Please try again tomorrow when quota resets.]"


# ── Backwards-compat shim so graph.py can still call get_llm() ───────────────
def get_llm():
    """
    Returns a lightweight callable shim so existing graph.py code that calls
    get_llm() continues to work without modification.
    The shim's .invoke(messages) delegates to llm_invoke_with_retry().
    """
    class _LLMShim:
        def invoke(self, messages):
            class _Response:
                def __init__(self, text):
                    self.content = text
            return _Response(llm_invoke_with_retry(messages))
    return _LLMShim()
