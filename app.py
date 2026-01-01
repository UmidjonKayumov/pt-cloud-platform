# app.py
# ==========================================================
# PT CLOUD PLATFORM (ISO/IEC 17043)
# Unified version: original PT logic + Role-Based Access + Audit Log
# Variant A: single-file architecture
# ==========================================================

from sqlalchemy import create_engine, text
import streamlit as st
import hashlib
import datetime
import pandas as pd

# ----------------------------------------------------------
# DATABASE CONNECTION (CLOUD READY)
# ----------------------------------------------------------

@st.cache_resource
def get_engine():
    return create_engine(
        st.secrets["postgres"]["url"],
        pool_pre_ping=True
    )



# ----------------------------------------------------------
# SECURITY UTILITIES
# ----------------------------------------------------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ----------------------------------------------------------
# DATABASE INITIALIZATION
# ----------------------------------------------------------

engine = get_engine()
with engine.begin() as conn:
    conn.execute(text("SELECT 1"))


            # USERS & ROLES
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # AUDIT LOG (ISO 17043)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    username TEXT,
                    action TEXT,
                    object_type TEXT,
                    object_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # PT PARTICIPANTS (FROM ORIGINAL SYSTEM)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pt_participants (
                    id SERIAL PRIMARY KEY,
                    lab_code TEXT,
                    lab_name TEXT,
                    scheme_code TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # PT RESULTS (PLACEHOLDER – YOUR ORIGINAL LOGIC GOES HERE)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pt_results (
                    id SERIAL PRIMARY KEY,
                    lab_code TEXT,
                    scheme_code TEXT,
                    reported_value NUMERIC,
                    z_score NUMERIC,
                    status TEXT,
                    submitted_by TEXT,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # CREATE SUPER ADMIN (CREATOR)
            cur.execute("SELECT COUNT(*) FROM users")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (%s,%s,%s)",
                    ("creator", hash_password("creator123"), "SUPER_ADMIN")
                )

        conn.commit()

# ----------------------------------------------------------
# AUDIT LOGGING
# ----------------------------------------------------------

def audit(username, action, object_type="", object_id=""):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (username, action, object_type, object_id) VALUES (%s,%s,%s,%s)",
                (username, action, object_type, object_id)
            )
        conn.commit()

# ----------------------------------------------------------
# AUTHENTICATION
# ----------------------------------------------------------

def authenticate(username, password):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role FROM users WHERE username=%s AND password_hash=%s AND is_active=TRUE",
                (username, hash_password(password))
            )
            row = cur.fetchone()
            return row[0] if row else None

# ----------------------------------------------------------
# ROLE GUARD
# ----------------------------------------------------------

def require_role(roles):
    if st.session_state.get("role") not in roles:
        st.error("⛔ Sizda bu bo‘limga ruxsat yo‘q")
        st.stop()

# ----------------------------------------------------------
# LOGIN UI
# ----------------------------------------------------------

def login_ui():
    st.title("PT Platform Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        role = authenticate(u, p)
        if role:
            st.session_state.user = u
            st.session_state.role = role
            audit(u, "LOGIN")
            st.rerun()
        else:
            st.error("Login yoki parol noto‘g‘ri")

# ----------------------------------------------------------
# USER MANAGEMENT (SUPER ADMIN ONLY)
# ----------------------------------------------------------

def user_management():
    require_role(["SUPER_ADMIN"])

    st.subheader("👤 Foydalanuvchilarni boshqarish")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    r = st.selectbox("Role", ["ADMIN", "OPERATOR", "VIEWER"])

    if st.button("Yaratish"):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (%s,%s,%s)",
                    (u, hash_password(p), r)
                )
            conn.commit()
        audit(st.session_state.user, "CREATE_USER", "USER", u)
        st.success("Foydalanuvchi yaratildi")

# ----------------------------------------------------------
# PT PARTICIPANTS UI (ADMIN / OPERATOR)
# ----------------------------------------------------------

def pt_participants_ui():
    require_role(["SUPER_ADMIN", "ADMIN", "OPERATOR"])

    st.subheader("🧪 PT Qatnashuvchilar")
    lab_code = st.text_input("Lab code")
    lab_name = st.text_input("Lab nomi")
    scheme = st.text_input("PT Scheme")

    if st.button("Saqlash"):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO pt_participants (lab_code, lab_name, scheme_code, created_by) VALUES (%s,%s,%s,%s)",
                    (lab_code, lab_name, scheme, st.session_state.user)
                )
            conn.commit()
        audit(st.session_state.user, "CREATE", "PT_PARTICIPANT", lab_code)
        st.success("Saqlandi")

# ----------------------------------------------------------
# AUDIT LOG VIEW
# ----------------------------------------------------------

def audit_log_ui():
    require_role(["SUPER_ADMIN", "ADMIN"])
    st.subheader("📜 Audit Log")

    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM audit_log ORDER BY timestamp DESC", conn)
    st.dataframe(df, use_container_width=True)

# ----------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------

def main():
    st.set_page_config("PT Cloud Platform", layout="wide")
    init_db()

    if "user" not in st.session_state:
        login_ui()
        return

    st.sidebar.success(f"👤 {st.session_state.user} ({st.session_state.role})")

    menu = ["PT Participants"]
    if st.session_state.role == "SUPER_ADMIN":
        menu += ["User Management", "Audit Log"]
    elif st.session_state.role == "ADMIN":
        menu += ["Audit Log"]

    choice = st.sidebar.radio("Menu", menu)

    if choice == "PT Participants":
        pt_participants_ui()
    elif choice == "User Management":
        user_management()
    elif choice == "Audit Log":
        audit_log_ui()

    if st.sidebar.button("Logout"):
        audit(st.session_state.user, "LOGOUT")
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()


