"""
db/models.py — Pydantic data models for type safety across the application.
"""

from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime
import re


class CustomerModel(BaseModel):
    customer_id: Optional[str] = None
    name:        str
    email:       str
    phone:       str
    created_at:  Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email address format.")
        return v.lower().strip()

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        digits = re.sub(r"[^\d]", "", v)
        if not (7 <= len(digits) <= 15):
            raise ValueError("Phone number must be between 7 and 15 digits.")
        return v.strip()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters.")
        return v.title()


class BookingModel(BaseModel):
    id:           Optional[str] = None
    customer_id:  str
    booking_type: str
    date:         str           # YYYY-MM-DD
    time:         str           # HH:MM
    status:       str = "confirmed"
    notes:        Optional[str] = ""
    created_at:   Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            d = datetime.strptime(v.strip(), "%Y-%m-%d")
            if d.date() < datetime.today().date():
                raise ValueError("Booking date cannot be in the past.")
            return v.strip()
        except ValueError as e:
            if "past" in str(e):
                raise
            raise ValueError("Date must be in YYYY-MM-DD format (e.g., 2025-08-15).")

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        try:
            datetime.strptime(v.strip(), "%H:%M")
            return v.strip()
        except ValueError:
            raise ValueError("Time must be in HH:MM format (e.g., 10:30).")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"confirmed", "cancelled", "completed", "pending"}
        if v.lower() not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v.lower()


class BookingPayload(BaseModel):
    """Input payload for the booking persistence tool."""
    name:         str
    email:        str
    phone:        str
    booking_type: str
    date:         str
    time:         str
    notes:        Optional[str] = ""
