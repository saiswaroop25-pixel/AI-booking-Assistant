"""
Chat Logic — intent detection, short-term memory management, and response routing.

Intents
-------
  BOOKING_INTENT   → user wants to book/schedule/reserve something
  RAG_INTENT       → user asks a question answerable from uploaded docs
  CANCEL_INTENT    → user wants to cancel an ongoing booking
  RETRIEVAL_INTENT → user wants to look up their existing booking
  GENERAL_INTENT   → general conversation / greeting / anything else

Short-term memory: last MAX_MEMORY_MESSAGES messages stored in st.session_state.
"""

import re
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.config import MAX_MEMORY_MESSAGES, BOOKING_DOMAIN, BOOKING_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection (keyword + LLM hybrid)
# ─────────────────────────────────────────────────────────────────────────────

BOOKING_KEYWORDS = [
    "book", "schedule", "appointment", "reserve", "slot", "visit",
    "consult", "consultation", "register", "sign up", "enroll",
    "i want to", "i'd like to", "i need to", "make an appointment",
    "fix an appointment", "set up", "arrange"
]

CANCEL_KEYWORDS = [
    "cancel", "stop", "quit", "abort", "exit", "nevermind", "never mind",
    "don't want", "do not want", "forget it", "skip"
]

RETRIEVAL_KEYWORDS = [
    "my booking", "check booking", "booking id", "find booking",
    "lookup", "look up", "booking status", "view booking", "show booking"
]


def detect_intent(user_message: str, llm=None, has_rag_docs: bool = False) -> str:
    """
    Detect user intent from their message.
    Uses keyword matching first; falls back to LLM if ambiguous.
    Returns one of: BOOKING_INTENT | CANCEL_INTENT | RETRIEVAL_INTENT | RAG_INTENT | GENERAL_INTENT
    """
    text = user_message.lower().strip()

    # Check cancel first (highest priority during active flows)
    if any(kw in text for kw in CANCEL_KEYWORDS):
        return "CANCEL_INTENT"

    # Retrieval intent
    if any(kw in text for kw in RETRIEVAL_KEYWORDS):
        return "RETRIEVAL_INTENT"

    # Booking intent
    booking_score = sum(1 for kw in BOOKING_KEYWORDS if kw in text)
    if booking_score >= 1:
        return "BOOKING_INTENT"

    # RAG intent: questions when docs are uploaded
    if has_rag_docs and _is_question(text):
        return "RAG_INTENT"

    # Fallback: use LLM if available
    if llm:
        try:
            prompt = f"""Classify this user message into ONE of these intents:
- BOOKING_INTENT: user wants to book, schedule, or make an appointment
- RAG_INTENT: user is asking a question about documents or information
- CANCEL_INTENT: user wants to cancel or stop
- GENERAL_INTENT: greeting, small talk, or anything else

User message: "{user_message}"

Reply with ONLY the intent label (no explanation)."""
            response = llm.invoke([HumanMessage(content=prompt)])
            detected = response.content.strip().upper()
            if detected in ["BOOKING_INTENT", "RAG_INTENT", "CANCEL_INTENT", "RETRIEVAL_INTENT", "GENERAL_INTENT"]:
                return detected
        except Exception:
            pass

    return "GENERAL_INTENT"


def _is_question(text: str) -> bool:
    """Heuristic: is the message a question?"""
    return (
        text.endswith("?")
        or text.startswith(("what", "how", "when", "where", "who", "why", "is ", "are ", "can ", "does ", "do "))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Short-term memory
# ─────────────────────────────────────────────────────────────────────────────

def get_memory(session_state) -> list:
    """Return the current conversation memory list."""
    return session_state.get("chat_memory", [])


def update_memory(session_state, role: str, content: str):
    """Append a message to memory, trimming to MAX_MEMORY_MESSAGES."""
    if "chat_memory" not in session_state:
        session_state["chat_memory"] = []
    session_state["chat_memory"].append({"role": role, "content": content})
    # Keep only the last N messages
    if len(session_state["chat_memory"]) > MAX_MEMORY_MESSAGES:
        session_state["chat_memory"] = session_state["chat_memory"][-MAX_MEMORY_MESSAGES:]


def clear_memory(session_state):
    session_state["chat_memory"] = []


def format_memory_for_llm(memory: list) -> list:
    """Convert memory list to LangChain message objects."""
    messages = []
    for msg in memory:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# General LLM response (non-booking, non-RAG)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a friendly and helpful AI Booking Assistant for {BOOKING_DOMAIN}.

Your capabilities:
1. Help users book medical appointments conversationally.
2. Answer questions using uploaded clinic documents (RAG).
3. Provide general information about the clinic's services.

Available appointment types: {', '.join(BOOKING_TYPES)}.

Be warm, concise, and professional. If the user seems to want to book an appointment,
gently guide them by saying something like "I can help you book an appointment!
Would you like to schedule one?"

Always respond in a helpful and empathetic manner appropriate for a healthcare setting.
"""


def get_general_response(llm, user_message: str, memory: list) -> str:
    """
    Generate a general conversational response using the LLM with memory context.
    """
    if llm is None:
        return (
            "I'm your AI Booking Assistant! I can help you schedule medical appointments "
            "or answer questions about our clinic. To get started, please configure your "
            "API key in the sidebar."
        )

    try:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(format_memory_for_llm(memory[-10:]))  # Last 10 for context
        messages.append(HumanMessage(content=user_message))

        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"I encountered an error processing your request: {str(e)}"


def get_llm_model(api_key: str):
    """Initialize the Groq LLM model, trying models in order until one works."""
    if not api_key:
        return None

    from langchain_groq import ChatGroq

    # Models in preference order — all currently active on Groq (March 2026)
    candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "gemma2-9b-it",
    ]

    for model_name in candidates:
        try:
            model = ChatGroq(api_key=api_key, model=model_name, temperature=0.3)
            # Lightweight probe — confirms the model is live
            model.invoke([HumanMessage(content="hi")])
            return model
        except Exception:
            continue  # try next model

    return None

