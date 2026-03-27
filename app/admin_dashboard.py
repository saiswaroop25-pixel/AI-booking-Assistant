"""
Admin Dashboard — mandatory admin UI to view, filter, search, and manage bookings.
Password-protected. Accessible from sidebar navigation.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from db.database import get_all_bookings, search_bookings, update_booking_status
from app.config import ADMIN_PASSWORD


def _check_auth() -> bool:
    """Return True if admin is authenticated."""
    return st.session_state.get("admin_authenticated", False)


def _login_screen():
    """Render the admin login form."""
    st.title("🔐 Admin Login")
    st.markdown("Please enter the admin password to access the dashboard.")

    with st.form("admin_login"):
        pwd = st.text_input("Password", type="password", placeholder="Enter admin password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if pwd == ADMIN_PASSWORD:
                st.session_state["admin_authenticated"] = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")

    st.info("💡 Default password: `admin123`  — change this in `app/config.py` before deploying.")


def _status_badge(status: str) -> str:
    """Return a colored emoji badge for booking status."""
    badges = {
        "confirmed":  "🟢 Confirmed",
        "cancelled":  "🔴 Cancelled",
        "completed":  "🔵 Completed",
        "pending":    "🟡 Pending",
    }
    return badges.get(status.lower(), f"⚪ {status.title()}")


def _metrics_row(bookings: list):
    """Display top-level metrics."""
    total = len(bookings)
    confirmed  = sum(1 for b in bookings if b["status"] == "confirmed")
    cancelled  = sum(1 for b in bookings if b["status"] == "cancelled")
    completed  = sum(1 for b in bookings if b["status"] == "completed")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Bookings", total)
    col2.metric("🟢 Confirmed",       confirmed)
    col3.metric("🔴 Cancelled",       cancelled)
    col4.metric("🔵 Completed",       completed)


def admin_dashboard_page():
    """Main admin dashboard page."""
    if not _check_auth():
        _login_screen()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_logout = st.columns([8, 1])
    with col_title:
        st.title("🏥 Admin Dashboard — Booking Management")
    with col_logout:
        if st.button("Logout", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.rerun()

    st.markdown("---")

    # ── Search & Filters ──────────────────────────────────────────────────────
    with st.expander("🔍 Search & Filter", expanded=True):
        col_search, col_status, col_date = st.columns([3, 2, 2])

        with col_search:
            search_query = st.text_input(
                "Search by name, email, or booking type",
                placeholder="e.g., John, john@email.com, Dental…",
                key="admin_search"
            )
        with col_status:
            status_filter = st.selectbox(
                "Filter by status",
                ["All", "confirmed", "cancelled", "completed"],
                key="admin_status"
            )
        with col_date:
            date_filter = st.date_input(
                "Filter by date (optional)",
                value=None,
                key="admin_date"
            )

    # ── Fetch data ────────────────────────────────────────────────────────────
    if search_query:
        bookings = search_bookings(search_query)
    else:
        bookings = get_all_bookings()

    # Apply status filter
    if status_filter != "All":
        bookings = [b for b in bookings if b["status"] == status_filter]

    # Apply date filter
    if date_filter:
        date_str = date_filter.strftime("%Y-%m-%d")
        bookings = [b for b in bookings if b["date"] == date_str]

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_bookings = get_all_bookings()
    _metrics_row(all_bookings)
    st.markdown("---")

    # ── Bookings table ────────────────────────────────────────────────────────
    st.subheader(f"📄 Bookings ({len(bookings)} results)")

    if not bookings:
        st.info("No bookings found. Bookings will appear here once customers start scheduling appointments.")
        return

    # Build display DataFrame
    rows = []
    for b in bookings:
        rows.append({
            "Booking ID":   b["id"],
            "Patient Name": b["name"],
            "Email":        b["email"],
            "Phone":        b["phone"],
            "Service":      b["booking_type"],
            "Date":         b["date"],
            "Time":         b["time"],
            "Status":       _status_badge(b["status"]),
            "Created At":   b["created_at"][:16] if b.get("created_at") else "—",
        })

    df = pd.DataFrame(rows)

    # Display table with column config
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "Booking ID": st.column_config.TextColumn("Booking ID", width="small"),
            "Status":     st.column_config.TextColumn("Status",     width="medium"),
        }
    )

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_export, col_spacer = st.columns([2, 5])
    with col_export:
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Export to CSV",
            data=csv,
            file_name=f"bookings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── Individual booking management ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Manage a Booking")

    with st.form("manage_booking"):
        col_id, col_new_status, col_btn = st.columns([3, 2, 2])
        with col_id:
            mgmt_id = st.text_input(
                "Booking ID",
                placeholder="e.g., A1B2C3D4",
                key="mgmt_booking_id"
            )
        with col_new_status:
            new_status = st.selectbox(
                "New Status",
                ["confirmed", "completed", "cancelled"],
                key="mgmt_new_status"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            update_btn = st.form_submit_button("Update Status", use_container_width=True)

        if update_btn:
            if not mgmt_id.strip():
                st.warning("Please enter a Booking ID.")
            else:
                success = update_booking_status(mgmt_id.strip().upper(), new_status)
                if success:
                    st.success(f"✅ Booking **{mgmt_id.upper()}** updated to **{new_status}**.")
                    st.rerun()
                else:
                    st.error(f"❌ Booking ID **{mgmt_id.upper()}** not found.")

    # ── Detailed booking view ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Booking Detail View")

    selected_id = st.selectbox(
        "Select a booking to view details",
        options=["— Select —"] + [b["id"] for b in bookings],
        key="detail_select"
    )

    if selected_id and selected_id != "— Select —":
        booking = next((b for b in bookings if b["id"] == selected_id), None)
        if booking:
            with st.container():
                st.markdown(f"""
<div style="background:#f0f4ff; border-left:4px solid #1a73e8; padding:16px; border-radius:8px;">

**🆔 Booking ID:** `{booking['id']}`  
**👤 Patient:** {booking['name']}  
**📧 Email:** {booking['email']}  
**📞 Phone:** {booking['phone']}  
**🏥 Service:** {booking['booking_type']}  
**📅 Date:** {booking['date']}  
**⏰ Time:** {booking['time']}  
**📌 Status:** {_status_badge(booking['status'])}  
**📝 Notes:** {booking.get('notes', '—') or '—'}  
**🕐 Created:** {booking.get('created_at', '—')}

</div>
""", unsafe_allow_html=True)
