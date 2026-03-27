"""
Booking Flow — multi-turn slot-filling state machine.

States
------
IDLE          → not in a booking flow
COLLECTING    → gathering required fields one by one
CONFIRMING    → all fields collected, awaiting user yes/no
COMPLETED     → booking saved, email sent
CANCELLED     → user cancelled mid-flow

Required slots: name, email, phone, booking_type, date, time
"""

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from app.config import BOOKING_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# State container
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_SLOTS = ["name", "email", "phone", "booking_type", "date", "time"]

SLOT_QUESTIONS = {
    "name":         "What is your full name?",
    "email":        "What is your email address?",
    "phone":        "What is your phone number?",
    "booking_type": (
        "What type of appointment would you like to book?\n"
        "Options: " + ", ".join(BOOKING_TYPES)
    ),
    "date":         "What date would you prefer? (Please use YYYY-MM-DD format, e.g., 2025-08-15)",
    "time":         "What time works best for you? (Please use HH:MM format, e.g., 10:30)",
}

SLOT_LABELS = {
    "name":         "Patient Name",
    "email":        "Email",
    "phone":        "Phone",
    "booking_type": "Appointment Type",
    "date":         "Date",
    "time":         "Time",
}

STATES = ["IDLE", "COLLECTING", "CONFIRMING", "COMPLETED", "CANCELLED"]


@dataclass
class BookingState:
    state: str = "IDLE"
    slots: Dict[str, str] = field(default_factory=dict)
    current_slot: Optional[str] = None
    booking_id: Optional[str] = None
    email_sent: bool = False
    retry_count: int = 0      # retry counter for current slot


def fresh_state() -> BookingState:
    return BookingState()


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_email(value: str) -> tuple[bool, str]:
    if re.match(r"[^@]+@[^@]+\.[^@]+", value.strip()):
        return True, ""
    return False, "That doesn't look like a valid email address. Please try again (e.g., user@example.com)."


def _validate_phone(value: str) -> tuple[bool, str]:
    digits = re.sub(r"[^\d]", "", value)
    if 7 <= len(digits) <= 15:
        return True, ""
    return False, "Please enter a valid phone number (7–15 digits)."


def _validate_date(value: str) -> tuple[bool, str]:
    try:
        d = datetime.strptime(value.strip(), "%Y-%m-%d")
        if d.date() < datetime.today().date():
            return False, "That date is in the past. Please enter a future date (YYYY-MM-DD)."
        return True, ""
    except ValueError:
        return False, "Please enter the date in YYYY-MM-DD format (e.g., 2025-08-20)."


def _validate_time(value: str) -> tuple[bool, str]:
    try:
        datetime.strptime(value.strip(), "%H:%M")
        return True, ""
    except ValueError:
        return False, "Please enter the time in HH:MM 24-hour format (e.g., 14:30 for 2:30 PM)."


def _validate_booking_type(value: str) -> tuple[bool, str]:
    # Fuzzy match: check if any booking type contains the value
    v = value.strip().lower()
    for bt in BOOKING_TYPES:
        if v in bt.lower() or bt.lower() in v:
            return True, ""
    # Check if it's a short number selection
    try:
        idx = int(v) - 1
        if 0 <= idx < len(BOOKING_TYPES):
            return True, ""
    except ValueError:
        pass
    return (
        False,
        f"Please choose one of: {', '.join(BOOKING_TYPES)}."
    )


def _normalize_booking_type(value: str) -> str:
    """Normalize booking type to the canonical name."""
    v = value.strip().lower()
    for bt in BOOKING_TYPES:
        if v in bt.lower() or bt.lower() in v:
            return bt
    try:
        idx = int(v) - 1
        if 0 <= idx < len(BOOKING_TYPES):
            return BOOKING_TYPES[idx]
    except ValueError:
        pass
    return value.strip().title()


VALIDATORS = {
    "email":        _validate_email,
    "phone":        _validate_phone,
    "date":         _validate_date,
    "time":         _validate_time,
    "booking_type": _validate_booking_type,
}

NORMALIZERS = {
    "booking_type": _normalize_booking_type,
    "date":         lambda v: v.strip(),
    "time":         lambda v: v.strip(),
    "email":        lambda v: v.strip().lower(),
    "phone":        lambda v: re.sub(r"[^\d+\-\s\(\)]", "", v).strip(),
    "name":         lambda v: v.strip().title(),
}


# ─────────────────────────────────────────────────────────────────────────────
# Pre-fill slots from the initial user message
# ─────────────────────────────────────────────────────────────────────────────

def _extract_slots_from_text(text: str) -> Dict[str, str]:
    """
    Try to pull known slot values from free-form text.
    E.g. "I want to book a dental checkup on 2025-08-20 at 10:00"
    """
    found = {}

    # Email
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    if m:
        found["email"] = m.group(0).lower()

    # Phone (10+ digit sequences, possibly with separators)
    m = re.search(r"[\+\d][\d\s\-\(\)]{8,}", text)
    if m:
        candidate = re.sub(r"[^\d]", "", m.group(0))
        if 7 <= len(candidate) <= 15:
            found["phone"] = m.group(0).strip()

    # Date YYYY-MM-DD
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if m:
        ok, _ = _validate_date(m.group(1))
        if ok:
            found["date"] = m.group(1)

    # Time HH:MM
    m = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if m:
        ok, _ = _validate_time(m.group(1).zfill(5))
        if ok:
            found["time"] = m.group(1).zfill(5)

    # Booking type
    text_l = text.lower()
    for bt in BOOKING_TYPES:
        if bt.lower() in text_l:
            found["booking_type"] = bt
            break

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Next missing slot
# ─────────────────────────────────────────────────────────────────────────────

def _next_missing_slot(slots: Dict[str, str]) -> Optional[str]:
    for s in REQUIRED_SLOTS:
        if not slots.get(s):
            return s
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation summary
# ─────────────────────────────────────────────────────────────────────────────

def build_confirmation_summary(slots: Dict[str, str]) -> str:
    lines = ["Here is a summary of your booking details:\n"]
    for slot in REQUIRED_SLOTS:
        label = SLOT_LABELS.get(slot, slot.title())
        value = slots.get(slot, "—")
        lines.append(f"• **{label}**: {value}")
    lines.append("\nShall I confirm this booking? Please reply **Yes** to confirm or **No** to cancel.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main flow processor
# ─────────────────────────────────────────────────────────────────────────────

def start_booking_flow(state: BookingState, initial_message: str) -> tuple[BookingState, str]:
    """
    Kick off a new booking flow.
    Pre-fills any slots already found in the initial message.
    Returns updated state and the bot's first response.
    """
    state = deepcopy(state)
    state.state = "COLLECTING"
    state.slots = {}
    state.booking_id = None
    state.email_sent = False
    state.retry_count = 0

    # Try to pre-fill from initial message
    prefilled = _extract_slots_from_text(initial_message)
    state.slots.update(prefilled)

    next_slot = _next_missing_slot(state.slots)
    if next_slot is None:
        # All slots already filled — move to confirmation
        state.state = "CONFIRMING"
        state.current_slot = None
        return state, build_confirmation_summary(state.slots)

    state.current_slot = next_slot
    greeting = "I'd be happy to help you book an appointment! 🏥\n\n"
    if prefilled:
        greeting += f"I've noted some details from your message. "
    return state, greeting + SLOT_QUESTIONS[next_slot]


def process_slot_input(state: BookingState, user_input: str) -> tuple[BookingState, str]:
    """
    Process user input during COLLECTING state.
    Validates the current slot, moves to the next one, or transitions to CONFIRMING.
    """
    state = deepcopy(state)
    slot = state.current_slot

    if slot is None:
        return state, "Something went wrong. Let me restart the booking. " + SLOT_QUESTIONS[REQUIRED_SLOTS[0]]

    value = user_input.strip()

    # Validate
    if slot in VALIDATORS:
        ok, error_msg = VALIDATORS[slot](value)
        if not ok:
            state.retry_count += 1
            if state.retry_count >= 3:
                state.state = "CANCELLED"
                return state, (
                    "It seems we're having trouble with that field. "
                    "Booking has been cancelled. Feel free to start again whenever you're ready!"
                )
            return state, f"⚠️ {error_msg}"

    # Normalize and store
    normalizer = NORMALIZERS.get(slot, lambda v: v.strip())
    state.slots[slot] = normalizer(value)
    state.retry_count = 0

    # Find next slot
    next_slot = _next_missing_slot(state.slots)
    if next_slot:
        state.current_slot = next_slot
        return state, SLOT_QUESTIONS[next_slot]
    else:
        # All collected → confirm
        state.state = "CONFIRMING"
        state.current_slot = None
        return state, build_confirmation_summary(state.slots)


def process_confirmation(state: BookingState, user_input: str) -> tuple[BookingState, str, Optional[dict]]:
    """
    Process yes/no confirmation.
    Returns (updated_state, bot_response, booking_payload_or_None).
    booking_payload is returned when the user confirmed → caller handles DB + email.
    """
    state = deepcopy(state)
    text = user_input.strip().lower()

    yes_words = {"yes", "y", "confirm", "confirmed", "ok", "okay", "sure", "yep", "yeah", "proceed", "go ahead"}
    no_words  = {"no", "n", "cancel", "cancelled", "stop", "nope", "nah", "abort", "nevermind", "never mind"}

    if any(w in text for w in yes_words):
        state.state = "COMPLETED"
        return state, "__CONFIRMED__", dict(state.slots)

    elif any(w in text for w in no_words):
        state.state = "CANCELLED"
        return state, "Your booking has been cancelled. No worries — feel free to start again whenever you're ready! 😊", None

    else:
        return state, (
            "I didn't quite understand. Please reply **Yes** to confirm your booking or **No** to cancel."
        ), None


def cancel_booking_flow(state: BookingState) -> tuple[BookingState, str]:
    state = deepcopy(state)
    state.state = "CANCELLED"
    state.slots = {}
    state.current_slot = None
    return state, "Booking process cancelled. Let me know if there's anything else I can help you with!"
