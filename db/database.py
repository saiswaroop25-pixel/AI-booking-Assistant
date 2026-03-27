"""
Database module — SQLite client for customers and bookings.
Schema:
  customers: customer_id (PK), name, email, phone
  bookings:  id (PK), customer_id (FK), booking_type, date, time, status, created_at
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Optional
import streamlit as st

DB_PATH = "bookings.db"


def get_connection():
    """Get SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()

    # Customers table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            email       TEXT NOT NULL UNIQUE,
            phone       TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Bookings table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id           TEXT PRIMARY KEY,
            customer_id  TEXT NOT NULL,
            booking_type TEXT NOT NULL,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL,
            status       TEXT DEFAULT 'confirmed',
            notes        TEXT DEFAULT '',
            created_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)

    conn.commit()
    conn.close()


def upsert_customer(name: str, email: str, phone: str) -> str:
    """
    Insert a new customer or return existing customer_id if email already exists.
    Returns the customer_id.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Check if customer already exists
    cur.execute("SELECT customer_id FROM customers WHERE email = ?", (email.lower().strip(),))
    row = cur.fetchone()

    if row:
        customer_id = row["customer_id"]
        # Update name/phone in case they changed
        cur.execute(
            "UPDATE customers SET name = ?, phone = ? WHERE customer_id = ?",
            (name, phone, customer_id)
        )
    else:
        customer_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO customers (customer_id, name, email, phone) VALUES (?, ?, ?, ?)",
            (customer_id, name, email.lower().strip(), phone)
        )

    conn.commit()
    conn.close()
    return customer_id


def create_booking(
    customer_id: str,
    booking_type: str,
    date: str,
    time: str,
    notes: str = ""
) -> str:
    """
    Insert a new booking record.
    Returns the booking ID.
    """
    conn = get_connection()
    cur = conn.cursor()

    booking_id = str(uuid.uuid4())[:8].upper()  # Short readable ID
    cur.execute(
        """
        INSERT INTO bookings (id, customer_id, booking_type, date, time, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (booking_id, customer_id, booking_type, date, time, notes)
    )
    conn.commit()
    conn.close()
    return booking_id


def get_all_bookings() -> list[dict]:
    """Fetch all bookings joined with customer info."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            b.id,
            c.name,
            c.email,
            c.phone,
            b.booking_type,
            b.date,
            b.time,
            b.status,
            b.notes,
            b.created_at
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        ORDER BY b.created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_booking_by_id(booking_id: str) -> Optional[dict]:
    """Fetch a single booking by its ID."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT b.*, c.name, c.email, c.phone
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        WHERE b.id = ?
    """, (booking_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_booking_status(booking_id: str, status: str) -> bool:
    """Update booking status (confirmed / cancelled / completed)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE bookings SET status = ? WHERE id = ?", (status, booking_id))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def search_bookings(query: str) -> list[dict]:
    """Search bookings by name, email, or date."""
    conn = get_connection()
    cur = conn.cursor()
    q = f"%{query}%"
    cur.execute("""
        SELECT
            b.id,
            c.name,
            c.email,
            c.phone,
            b.booking_type,
            b.date,
            b.time,
            b.status,
            b.notes,
            b.created_at
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        WHERE c.name LIKE ? OR c.email LIKE ? OR b.date LIKE ? OR b.booking_type LIKE ?
        ORDER BY b.created_at DESC
    """, (q, q, q, q))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
