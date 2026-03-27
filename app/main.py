"""
main.py — Streamlit entry point for the AI Booking Assistant.

Pages
-----
  💬 Chat          → Main booking chatbot + RAG
  📊 Admin         → Admin dashboard (password protected)
  📖 Instructions  → Setup & usage guide

Run with:
  streamlit run app/main.py
"""

import os
import sys
import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.database import init_db
from app.chat_logic import (
    detect_intent,
    get_general_response,
    get_llm_model,
    get_memory,
    update_memory,
    clear_memory,
)
from app.booking_flow import (
    BookingState,
    fresh_state,
    start_booking_flow,
    process_slot_input,
    process_confirmation,
    cancel_booking_flow,
)
from app.rag_pipeline import VectorStore, extract_text_from_pdf, chunk_text
from app.tools import (
    rag_tool,
    booking_persistence_tool,
    send_booking_confirmation_email,
)
from app.admin_dashboard import admin_dashboard_page
from app.config import BOOKING_DOMAIN


# ─────────────────────────────────────────────────────────────────────────────
# App-wide init
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    """Initialize all session state keys once."""
    defaults = {
        "messages":           [],          # display messages [{role, content}]
        "chat_memory":        [],          # short-term memory for LLM
        "booking_state":      fresh_state(),
        "vector_store":       VectorStore(),
        "processed_pdfs":     [],          # names of already-processed PDFs
        "llm":                None,
        "api_key_set":        False,
        "admin_authenticated": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/clinic.png", width=64)
        st.title(f"AI Booking\nAssistant")
        st.caption(BOOKING_DOMAIN)
        st.markdown("---")

        # Page navigation
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📊 Admin Dashboard", "📖 Instructions"],
            index=0,
            key="nav_page"
        )

        st.markdown("---")

        # API Key input
        st.subheader("🔑 Groq API Key")
        api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            placeholder="gsk_...",
            value=st.session_state.get("_api_key_input", ""),
            key="_api_key_input",
            help="Get your free key at https://console.groq.com"
        )

        if st.button("Connect LLM", use_container_width=True, type="primary"):
            if api_key.strip():
                with st.spinner("Connecting to Groq..."):
                    llm = get_llm_model(api_key.strip())
                    if llm:
                        st.session_state["llm"] = llm
                        st.session_state["api_key_set"] = True
                        st.success("✅ LLM connected!")
                    else:
                        st.error("❌ Could not connect. Check your API key.")
            else:
                st.warning("Please enter an API key.")

        if st.session_state.get("api_key_set"):
            st.success("🟢 LLM Active")
        else:
            st.warning("🔴 LLM Not Connected")

        st.markdown("---")

        # PDF Upload for RAG
        st.subheader("📄 Upload Clinic Documents")
        st.caption("Upload PDFs for RAG-based Q&A")

        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed"
        )

        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state["processed_pdfs"]
            ]
            if new_files:
                vs = st.session_state["vector_store"]
                for pdf_file in new_files:
                    with st.spinner(f"Processing {pdf_file.name}…"):
                        try:
                            raw_bytes = pdf_file.read()
                            text = extract_text_from_pdf(raw_bytes)
                            if not text or len(text) < 50:
                                st.warning(f"⚠️ Could not extract text from {pdf_file.name}.")
                                continue
                            chunks = chunk_text(text)
                            ok = vs.add_documents(chunks)
                            if ok:
                                st.session_state["processed_pdfs"].append(pdf_file.name)
                                st.success(f"✅ {pdf_file.name} ({len(chunks)} chunks)")
                            else:
                                st.error(f"❌ Failed to embed {pdf_file.name}.")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                pass  # already processed

        if st.session_state["vector_store"].is_ready():
            n = st.session_state["vector_store"].chunk_count()
            st.info(f"📚 {n} chunks indexed from {len(st.session_state['processed_pdfs'])} file(s)")

        st.markdown("---")

        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                clear_memory(st.session_state)
                st.session_state["booking_state"] = fresh_state()
                st.rerun()
        with col2:
            if st.button("📄 Clear Docs", use_container_width=True):
                st.session_state["vector_store"] = VectorStore()
                st.session_state["processed_pdfs"] = []
                st.rerun()

    return page


# ─────────────────────────────────────────────────────────────────────────────
# Chat page
# ─────────────────────────────────────────────────────────────────────────────

def _add_message(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content})
    update_memory(st.session_state, role, content)


def _bot_reply(content: str):
    _add_message("assistant", content)


def _process_user_message(user_input: str):
    """
    Core message processing pipeline:
    1. If inside booking flow → route to booking_flow handlers
    2. Else → detect intent → route to appropriate handler
    """
    _add_message("user", user_input)

    llm          = st.session_state.get("llm")
    booking_state: BookingState = st.session_state["booking_state"]
    vector_store: VectorStore   = st.session_state["vector_store"]
    memory = get_memory(st.session_state)

    # ── Active booking flow ───────────────────────────────────────────────────
    if booking_state.state == "COLLECTING":
        # Check for cancellation first
        cancel_words = {"cancel", "stop", "quit", "abort", "exit", "nevermind"}
        if any(w in user_input.lower() for w in cancel_words):
            new_state, response = cancel_booking_flow(booking_state)
            st.session_state["booking_state"] = new_state
            _bot_reply(response)
            return

        new_state, response = process_slot_input(booking_state, user_input)
        st.session_state["booking_state"] = new_state
        _bot_reply(response)
        return

    if booking_state.state == "CONFIRMING":
        new_state, response, payload = process_confirmation(booking_state, user_input)
        st.session_state["booking_state"] = new_state

        if response == "__CONFIRMED__" and payload:
            # Save to DB
            result = booking_persistence_tool(payload)
            if result["success"]:
                booking_id = result["booking_id"]
                new_state.booking_id = booking_id

                # Send email
                email_result = send_booking_confirmation_email(
                    name=payload["name"],
                    email=payload["email"],
                    booking_id=booking_id,
                    booking_type=payload["booking_type"],
                    date=payload["date"],
                    time=payload["time"],
                    notes=payload.get("notes", ""),
                )

                # Build confirmation message
                email_note = (
                    f"\n\n📧 *{email_result['message']}*"
                    if not email_result["success"]
                    else f"\n\n📧 A confirmation email has been sent to **{payload['email']}**."
                )
                confirm_msg = (
                    f"🎉 **Booking Confirmed!**\n\n"
                    f"Your appointment has been saved.\n\n"
                    f"**Booking ID:** `{booking_id}`\n"
                    f"**Patient:** {payload['name']}\n"
                    f"**Service:** {payload['booking_type']}\n"
                    f"**Date:** {payload['date']}\n"
                    f"**Time:** {payload['time']}"
                    f"{email_note}\n\n"
                    f"Is there anything else I can help you with?"
                )
                _bot_reply(confirm_msg)
            else:
                _bot_reply(
                    f"⚠️ There was an issue saving your booking: {result['message']}\n\n"
                    f"Please try again or contact us directly."
                )
            st.session_state["booking_state"] = new_state
        else:
            _bot_reply(response)
        return

    # ── Intent detection ──────────────────────────────────────────────────────
    intent = detect_intent(
        user_message=user_input,
        llm=llm,
        has_rag_docs=vector_store.is_ready()
    )

    if intent == "BOOKING_INTENT":
        new_state, response = start_booking_flow(fresh_state(), user_input)
        st.session_state["booking_state"] = new_state
        _bot_reply(response)

    elif intent == "RAG_INTENT":
        response = rag_tool(user_input, vector_store, llm)
        _bot_reply(response)

    elif intent == "RETRIEVAL_INTENT":
        # Try to extract a booking ID from the message
        import re
        m = re.search(r"\b([A-Z0-9]{6,10})\b", user_input.upper())
        if m:
            from db.database import get_booking_by_id
            booking = get_booking_by_id(m.group(1))
            if booking:
                _bot_reply(
                    f"📋 **Booking Found**\n\n"
                    f"**ID:** `{booking['id']}`\n"
                    f"**Patient:** {booking['name']}\n"
                    f"**Service:** {booking['booking_type']}\n"
                    f"**Date:** {booking['date']} at {booking['time']}\n"
                    f"**Status:** {booking['status'].title()}"
                )
            else:
                _bot_reply(f"I couldn't find a booking with ID **{m.group(1)}**. Please double-check the ID.")
        else:
            _bot_reply(
                "To look up your booking, please provide your **Booking ID** "
                "(e.g., 'Check booking A1B2C3D4')."
            )

    else:
        # General conversation
        response = get_general_response(llm, user_input, memory)
        _bot_reply(response)


def chat_page():
    st.title("💬 AI Booking Assistant")
    st.caption(f"🏥 {BOOKING_DOMAIN} — Book appointments, ask questions, get help")

    # Status bar
    status_cols = st.columns(3)
    with status_cols[0]:
        if st.session_state.get("api_key_set"):
            st.success("🟢 LLM Ready")
        else:
            st.warning("🔴 LLM Not Connected — enter API key in sidebar")
    with status_cols[1]:
        if st.session_state["vector_store"].is_ready():
            st.success(f"📚 {st.session_state['vector_store'].chunk_count()} chunks indexed")
        else:
            st.info("📄 No docs uploaded — upload PDFs for Q&A")
    with status_cols[2]:
        bs = st.session_state["booking_state"]
        state_map = {
            "IDLE":       "💤 No active booking",
            "COLLECTING": "📝 Collecting booking details",
            "CONFIRMING": "⏳ Awaiting confirmation",
            "COMPLETED":  "✅ Booking complete",
            "CANCELLED":  "❌ Booking cancelled",
        }
        st.info(state_map.get(bs.state, bs.state))

    st.markdown("---")

    # Welcome message on first load
    if not st.session_state["messages"]:
        welcome = (
            f"👋 **Welcome to the {BOOKING_DOMAIN} AI Booking Assistant!**\n\n"
            "I can help you:\n"
            "- 📅 **Book an appointment** — just say *'I'd like to book an appointment'*\n"
            "- 🔍 **Answer questions** from uploaded clinic documents\n"
            "- 🔎 **Look up a booking** by ID\n\n"
            "How can I assist you today?"
        )
        st.session_state["messages"].append({"role": "assistant", "content": welcome})

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"], avatar="🏥" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Type your message here…", key="chat_input"):
        with st.spinner("Thinking…"):
            _process_user_message(user_input)
        st.rerun()

    # Quick action buttons
    st.markdown("---")
    st.caption("Quick actions:")
    qcols = st.columns(4)
    quick_actions = [
        ("📅 Book Appointment",    "I'd like to book an appointment"),
        ("🔎 Check My Booking",    "I want to check my booking"),
        ("❓ What services?",      "What services do you offer?"),
        ("🕒 Opening hours?",     "What are your opening hours?"),
    ]
    for col, (label, msg) in zip(qcols, quick_actions):
        if col.button(label, use_container_width=True):
            with st.spinner("Thinking…"):
                _process_user_message(msg)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Instructions page
# ─────────────────────────────────────────────────────────────────────────────

def instructions_page():
    st.title("📖 Setup & Usage Guide")
    st.markdown("---")

    st.markdown("""
## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your API key
Get a **free** Groq API key at [console.groq.com](https://console.groq.com) and enter it in the sidebar.

### 3. (Optional) Configure email
Set up SMTP for confirmation emails. Either:
- Edit `.streamlit/secrets.toml` with your credentials, **or**
- Set environment variables: `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`

For Gmail, use an [App Password](https://myaccount.google.com/apppasswords) (not your regular password).

```toml
# .streamlit/secrets.toml
GROQ_API_KEY   = "gsk_..."
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = "587"
SMTP_USER      = "youraddress@gmail.com"
SMTP_PASSWORD  = "your-app-password"
```

### 4. Run the app
```bash
streamlit run app/main.py
```

---

## 💡 Features

| Feature | Description |
|---------|-------------|
| 🤖 RAG Chatbot | Upload clinic PDFs → ask questions → get accurate answers |
| 📅 Booking Flow | Conversational multi-turn appointment booking |
| 🗄️ Database | SQLite-backed storage (customers + bookings) |
| 📧 Email | Automatic HTML confirmation emails after booking |
| 📊 Admin Dashboard | View, search, filter, update, and export all bookings |
| 🧠 Memory | Remembers last 25 messages for context continuity |

---

## 📁 Project Structure

```
ai_booking_assistant/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── chat_logic.py        # Intent detection + memory
│   ├── booking_flow.py      # Slot-filling state machine
│   ├── rag_pipeline.py      # PDF processing + FAISS
│   ├── tools.py             # RAG / DB / Email tools
│   ├── admin_dashboard.py   # Admin UI
│   └── config.py            # Configuration
├── db/
│   └── database.py          # SQLite client
├── .streamlit/
│   └── secrets.toml         # API keys (git-ignored)
├── requirements.txt
└── README.md
```

---

## 🔐 Admin Dashboard
- Navigate to **📊 Admin Dashboard** in the sidebar
- Default password: `admin123` (change in `app/config.py`)
- Features: view all bookings, search/filter, update status, export CSV

---

## 🌐 Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Set main file: `app/main.py`
4. Add secrets in the Streamlit Cloud dashboard under **Settings → Secrets**
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Booking Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": f"AI Booking Assistant — {BOOKING_DOMAIN}"
        }
    )

    # Initialize DB and session state
    init_db()
    _init_session()

    # Render sidebar and get current page
    page = _sidebar()

    # Route to page
    if page == "💬 Chat":
        chat_page()
    elif page == "📊 Admin Dashboard":
        admin_dashboard_page()
    elif page == "📖 Instructions":
        instructions_page()


if __name__ == "__main__":
    main()
