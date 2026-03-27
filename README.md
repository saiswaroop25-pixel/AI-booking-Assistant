# 🏥 AI Booking Assistant

An AI-driven Booking Assistant for a **Medical / Healthcare Clinic**, built with Streamlit, LangChain, Groq LLM, FAISS, and SQLite.

---

## ✨ Features

| Feature | Details |
|---|---|
| 💬 **RAG Chatbot** | Upload clinic PDFs → answer patient questions from document content |
| 📅 **Conversational Booking** | Multi-turn slot-filling to collect name, email, phone, service, date, time |
| ✅ **Confirmation Flow** | Summarises details, asks for explicit yes/no before saving |
| 🗄️ **SQLite Database** | Persistent storage for `customers` and `bookings` tables |
| 📧 **Email Confirmation** | HTML confirmation email sent via SMTP after each booking |
| 📊 **Admin Dashboard** | Password-protected view with search, filter, status update, CSV export |
| 🧠 **Short-Term Memory** | Retains last 25 messages for context-aware conversations |
| ⚠️ **Error Handling** | Validates email, date format, phone, past dates; friendly error messages |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/ai-booking-assistant.git
cd ai-booking-assistant
pip install -r requirements.txt
```

### 2. Add your API key

Create `.streamlit/secrets.toml` (copy from the template):

```toml
GROQ_API_KEY  = "gsk_..."          # https://console.groq.com (free)
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = "587"
SMTP_USER     = "you@gmail.com"
SMTP_PASSWORD = "your-app-password"   # Gmail App Password, not your main password
```

> **Email is optional** — if not configured, the booking is still saved and the user is notified.

### 3. Run

```bash
streamlit run app/main.py
```

---

## 📁 Project Structure

```
ai_booking_assistant/
├── app/
│   ├── main.py              # Streamlit entry point — pages, sidebar, routing
│   ├── chat_logic.py        # Intent detection (BOOKING / RAG / GENERAL)
│   │                        # Short-term memory management
│   ├── booking_flow.py      # Slot-filling state machine
│   │                        # (IDLE → COLLECTING → CONFIRMING → COMPLETED)
│   ├── rag_pipeline.py      # PDF text extraction, chunking, FAISS vector store
│   ├── tools.py             # Tool implementations:
│   │                        #   rag_tool, booking_persistence_tool, email_tool
│   ├── admin_dashboard.py   # Admin UI (password-protected)
│   └── config.py            # Centralised settings + API key access
├── db/
│   ├── database.py          # SQLite client (init, upsert, search, update)
│   └── models.py            # Pydantic models for type safety
├── docs/                    # Place sample clinic PDFs here
├── .streamlit/
│   └── secrets.toml         # ← git-ignored, fill in your credentials
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Architecture

```
User Message
     │
     ▼
Intent Detection (keyword + LLM)
     │
     ├─ BOOKING_INTENT ──► Booking Flow State Machine
     │                        COLLECTING → validate slots one-by-one
     │                        CONFIRMING → ask yes/no
     │                        CONFIRMED  → DB save + email
     │
     ├─ RAG_INTENT ──────► FAISS similarity search → LLM blending → answer
     │
     ├─ RETRIEVAL_INTENT ─► DB lookup by booking ID
     │
     └─ GENERAL_INTENT ──► LLM with system prompt + memory context
```

---

## 📊 Database Schema

```sql
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    phone       TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE bookings (
    id           TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
    booking_type TEXT NOT NULL,
    date         TEXT NOT NULL,   -- YYYY-MM-DD
    time         TEXT NOT NULL,   -- HH:MM
    status       TEXT DEFAULT 'confirmed',
    notes        TEXT DEFAULT '',
    created_at   TEXT DEFAULT (datetime('now'))
);
```

---

## 🔒 Admin Dashboard

- Navigate to **📊 Admin Dashboard** in the sidebar
- Default password: `admin123` — **change this** in `app/config.py` before deploying
- Features: view all bookings, search by name/email/date, filter by status, update booking status, export to CSV

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `.streamlit/secrets.toml` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Set **Main file path**: `app/main.py`
4. Go to **Settings → Secrets** and paste your `secrets.toml` contents there

> Note: SQLite data resets when Streamlit Cloud restarts the app. For persistence, swap the DB for **Supabase** (just replace `database.py` calls with the Supabase Python client).

---

## 🛠️ Customisation

| What to change | Where |
|---|---|
| Booking domain / clinic name | `app/config.py` → `BOOKING_DOMAIN` |
| Available service types | `app/config.py` → `BOOKING_TYPES` |
| LLM model | `app/config.py` → `GROQ_MODEL` |
| Admin password | `app/config.py` → `ADMIN_PASSWORD` |
| Embedding model | `app/config.py` → `EMBEDDING_MODEL` |
| Chunk size | `app/config.py` → `CHUNK_SIZE`, `CHUNK_OVERLAP` |

---

## 📦 Dependencies

- **streamlit** — UI framework
- **langchain / langchain-groq** — LLM orchestration
- **sentence-transformers** — local embeddings (no API cost)
- **faiss-cpu** — vector similarity search
- **pypdf** — PDF text extraction
- **pydantic** — data validation

---

## 📝 Notes

- The embedding model (`all-MiniLM-L6-v2`) runs **locally** — no API key needed for RAG.
- SQLite is used for simplicity; swap for Supabase by updating `db/database.py`.
- Email sending is **optional** — if SMTP is not configured, bookings are still saved.
