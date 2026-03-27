"""
Configuration module for AI Booking Assistant.
All API keys and settings are loaded from Streamlit secrets or environment variables.
"""

import os
import streamlit as st


def get_groq_api_key() -> str:
    """Get Groq API key from secrets or environment."""
    try:
        return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


def get_openai_api_key() -> str:
    """Get OpenAI API key from secrets or environment (used for embeddings)."""
    try:
        return st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def get_smtp_config() -> dict:
    """Get SMTP configuration for email sending."""
    try:
        return {
            "host": st.secrets.get("SMTP_HOST", os.getenv("SMTP_HOST", "smtp.gmail.com")),
            "port": int(st.secrets.get("SMTP_PORT", os.getenv("SMTP_PORT", "587"))),
            "user": st.secrets.get("SMTP_USER", os.getenv("SMTP_USER", "")),
            "password": st.secrets.get("SMTP_PASSWORD", os.getenv("SMTP_PASSWORD", "")),
        }
    except Exception:
        return {
            "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "user": os.getenv("SMTP_USER", ""),
            "password": os.getenv("SMTP_PASSWORD", ""),
        }


# LLM Model settings
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local sentence-transformers model

# Chunking settings for RAG
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Memory settings
MAX_MEMORY_MESSAGES = 25

# Booking domain
BOOKING_DOMAIN = "Medical / Healthcare Clinic"
BOOKING_TYPES = [
    "General Consultation",
    "Specialist Appointment",
    "Dental Checkup",
    "Eye Examination",
    "Physiotherapy Session",
    "Lab Tests",
    "Follow-up Visit",
    "Emergency Consultation",
]

# Database path
DB_PATH = "bookings.db"

# Admin password (change in production)
ADMIN_PASSWORD = "admin123"
