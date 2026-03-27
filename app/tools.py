"""
Tools module — implements the 3 core tools:
  1. RAG Tool        : query → retrieved answer from uploaded PDFs
  2. Booking Tool    : structured payload → saves to DB, returns booking ID
  3. Email Tool      : to/subject/body → sends confirmation email
"""

import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

from app.config import get_smtp_config, BOOKING_DOMAIN
from app.rag_pipeline import VectorStore, build_rag_context
from db.database import upsert_customer, create_booking, get_booking_by_id


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAG TOOL
# ─────────────────────────────────────────────────────────────────────────────

def rag_tool(query: str, vector_store: VectorStore, llm) -> str:
    """
    Input : user query string
    Output: answer blended from retrieved PDF chunks + LLM generation
    Returns a plain string answer.
    """
    if not vector_store.is_ready():
        return (
            "No documents have been uploaded yet. "
            "Please upload a PDF using the sidebar to enable document-based Q&A."
        )

    context = build_rag_context(vector_store, query)

    if not context:
        return "I couldn't find relevant information in the uploaded documents for that question."

    rag_prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information in the uploaded documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=rag_prompt)])
        return response.content.strip()
    except Exception as e:
        return f"Error generating RAG answer: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. BOOKING PERSISTENCE TOOL
# ─────────────────────────────────────────────────────────────────────────────

def booking_persistence_tool(booking_payload: dict) -> dict:
    """
    Input : structured booking payload dict with keys:
              name, email, phone, booking_type, date, time, notes (optional)
    Output: {"success": bool, "booking_id": str, "message": str}
    """
    required = ["name", "email", "phone", "booking_type", "date", "time"]
    for field in required:
        if not booking_payload.get(field):
            return {
                "success": False,
                "booking_id": None,
                "message": f"Missing required field: {field}"
            }

    # Validate email
    if not re.match(r"[^@]+@[^@]+\.[^@]+", booking_payload["email"]):
        return {
            "success": False,
            "booking_id": None,
            "message": "Invalid email format. Please enter a valid email address."
        }

    # Validate date (YYYY-MM-DD)
    try:
        booking_date = datetime.strptime(booking_payload["date"], "%Y-%m-%d")
        if booking_date.date() < datetime.today().date():
            return {
                "success": False,
                "booking_id": None,
                "message": "Booking date cannot be in the past. Please choose a future date."
            }
    except ValueError:
        return {
            "success": False,
            "booking_id": None,
            "message": "Invalid date format. Please use YYYY-MM-DD (e.g., 2025-08-15)."
        }

    # Validate time (HH:MM)
    try:
        datetime.strptime(booking_payload["time"], "%H:%M")
    except ValueError:
        return {
            "success": False,
            "booking_id": None,
            "message": "Invalid time format. Please use HH:MM (e.g., 10:30)."
        }

    try:
        customer_id = upsert_customer(
            name=booking_payload["name"].strip(),
            email=booking_payload["email"].strip().lower(),
            phone=booking_payload["phone"].strip(),
        )
        booking_id = create_booking(
            customer_id=customer_id,
            booking_type=booking_payload["booking_type"],
            date=booking_payload["date"],
            time=booking_payload["time"],
            notes=booking_payload.get("notes", ""),
        )
        return {
            "success": True,
            "booking_id": booking_id,
            "message": f"Booking saved successfully with ID: {booking_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "booking_id": None,
            "message": f"Database error while saving booking: {str(e)}"
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMAIL TOOL
# ─────────────────────────────────────────────────────────────────────────────

def _build_confirmation_html(
    name: str,
    booking_id: str,
    booking_type: str,
    date: str,
    time: str,
    domain: str = BOOKING_DOMAIN,
    notes: str = "",
) -> str:
    """Build a nice HTML email body for booking confirmation."""
    notes_section = f"<p><strong>Notes:</strong> {notes}</p>" if notes else ""
    return f"""
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
  <div style="background: #1a73e8; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
    <h1 style="margin:0;">✅ Booking Confirmed</h1>
    <p style="margin:4px 0 0 0; opacity:0.9;">{domain}</p>
  </div>
  <div style="background: #f9f9f9; padding: 24px; border: 1px solid #e0e0e0; border-radius: 0 0 8px 8px;">
    <p>Dear <strong>{name}</strong>,</p>
    <p>Your appointment has been successfully booked. Here are your details:</p>
    <table style="width:100%; border-collapse: collapse; margin: 16px 0;">
      <tr style="background:#e8f0fe;">
        <td style="padding:10px; font-weight:bold; border:1px solid #c5cae9;">Booking ID</td>
        <td style="padding:10px; border:1px solid #c5cae9; font-family:monospace; font-size:16px; color:#1a73e8;">{booking_id}</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:bold; border:1px solid #c5cae9;">Service</td>
        <td style="padding:10px; border:1px solid #c5cae9;">{booking_type}</td>
      </tr>
      <tr style="background:#e8f0fe;">
        <td style="padding:10px; font-weight:bold; border:1px solid #c5cae9;">Date</td>
        <td style="padding:10px; border:1px solid #c5cae9;">{date}</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:bold; border:1px solid #c5cae9;">Time</td>
        <td style="padding:10px; border:1px solid #c5cae9;">{time}</td>
      </tr>
    </table>
    {notes_section}
    <div style="background:#fff3cd; border:1px solid #ffc107; padding:12px; border-radius:6px; margin-top:16px;">
      <strong>📌 Reminder:</strong> Please arrive 10 minutes early. Bring a valid ID and any previous medical records if applicable.
    </div>
    <p style="margin-top:24px; color:#555;">
      If you need to reschedule or cancel, please contact us with your Booking ID.
    </p>
    <p style="color:#888; font-size:12px; margin-top:32px;">
      This is an automated confirmation email. Please do not reply directly to this message.
    </p>
  </div>
</body>
</html>
"""


def email_tool(
    to_email: str,
    subject: str,
    body_html: str,
    body_text: str = "",
) -> dict:
    """
    Input : to_email, subject, body_html (and optional plain-text fallback)
    Output: {"success": bool, "message": str}
    """
    smtp = get_smtp_config()

    if not smtp["user"] or not smtp["password"]:
        return {
            "success": False,
            "message": (
                "Email could not be sent (SMTP credentials not configured), "
                "but your booking was saved successfully."
            )
        }

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp["user"]
        msg["To"] = to_email

        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(smtp["host"], smtp["port"], timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp["user"], smtp["password"])
            server.sendmail(smtp["user"], to_email, msg.as_string())

        return {"success": True, "message": f"Confirmation email sent to {to_email}."}

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": "Email could not be sent (authentication failed), but booking was saved."
        }
    except smtplib.SMTPException as e:
        return {
            "success": False,
            "message": f"Email could not be sent ({str(e)}), but booking was saved."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Email delivery failed: {str(e)}. Booking was saved successfully."
        }


def send_booking_confirmation_email(
    name: str,
    email: str,
    booking_id: str,
    booking_type: str,
    date: str,
    time: str,
    notes: str = "",
) -> dict:
    """High-level helper: build + send a booking confirmation email."""
    subject = f"Booking Confirmed — {booking_type} on {date} at {time} [ID: {booking_id}]"
    html = _build_confirmation_html(
        name=name,
        booking_id=booking_id,
        booking_type=booking_type,
        date=date,
        time=time,
        notes=notes,
    )
    plain = (
        f"Dear {name},\n\n"
        f"Your booking is confirmed!\n\n"
        f"Booking ID : {booking_id}\n"
        f"Service    : {booking_type}\n"
        f"Date       : {date}\n"
        f"Time       : {time}\n"
        f"{'Notes      : ' + notes if notes else ''}\n\n"
        f"Please arrive 10 minutes early and bring a valid ID.\n\n"
        f"Thank you for choosing us!"
    )
    return email_tool(to_email=email, subject=subject, body_html=html, body_text=plain)
