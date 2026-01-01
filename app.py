# app.py
# --------------------------------------------
# PT Suite – Proficiency Testing Platform
# Sistemalashgan, modulga bo'lingan Streamlit dasturi
# PostgreSQL (Neon) bilan ishlaydigan yakuniy versiya
# --------------------------------------------

import psycopg2
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
from pathlib import Path
from datetime import datetime

# ============================================
# 1. GLOBAL SOZLAMALAR VA LOGOTIP
# ============================================

APP_TITLE = "PT Suite – Proficiency Testing Platform"

# Neon URL uchun fallback constant (agar st.secrets bo'lmasa)
# ⚠️ Bu URL'ni keyin GitHub'ga qo'ymaganing ma'qul, lekin hozir lokalda ishlashi uchun qoldiryapman.
POSTGRES_URL = (
    "postgresql://neondb_owner:"
    "npg_4Xog9qaNzUEZ"
    "@ep-mute-leaf-ag5jfjwl-pooler.c-2.eu-central-1.aws.neon.tech/"
    "neondb?sslmode=require&channel_binding=require"
)

DATA_DIR = Path("pt_suite_data")
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✅",
    layout="wide"
)


def render_logo_sidebar():
    """PT Suite logotipi va kichik taglayn."""
    svg = """
    <svg width="210" height="60" viewBox="0 0 210 60" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#06b6d4;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#22c55e;stop-opacity:1" />
        </linearGradient>
      </defs>
      <rect x="2" y="5" rx="15" ry="15" width="120" height="50" fill="url(#grad1)" />
      <text x="20" y="38" font-family="Segoe UI, sans-serif" font-size="24" fill="white" font-weight="bold">PT</text>
      <text x="55" y="38" font-family="Segoe UI, sans-serif" font-size="18" fill="white">Suite</text>
      <circle cx="150" cy="30" r="15" fill="none" stroke="#06b6d4" stroke-width="3" />
      <polyline points="142,30 148,36 160,22" fill="none" stroke="#22c55e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    """
    st.sidebar.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 0.5rem;">
            {svg}
            <div style="font-family:Segoe UI, sans-serif; font-size:0.80rem; color:#4b5563; margin-top:0.3rem;">
                Proficiency Testing Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================
# 2. FOYDALI FUNKSIYALAR (UTILS)
# ============================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def check_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def round_2(x):
    """Kamida 2 ta ahamiyatli kasr bilan formatlash uchun."""
    if pd.isna(x):
        return ""
    return f"{x:.2f}"


def dataframe_to_excel_bytes(df, sheet_name="Report"):
    """DataFrame ni Excel faylga aylantirib, bytes ko‘rinishda qaytaradi."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


# ============================================
# 3. MA’LUMOTLAR QATLAMI (DATABASE LAYER)
# ============================================

def _resolve_db_url() -> str:
    """
    DB URL'ni topish:
    1) st.secrets['postgres']['url'] bo'lsa – shuni oladi
    2) bo'lmasa POSTGRES_URL constantiga tushadi
    """
    # 1. Secrets.toml orqali (Streamlit usuli)
    try:
        if "postgres" in st.secrets and "url" in st.secrets["postgres"]:
            return st.secrets["postgres"]["url"]
    except Exception:
        # Lokal ishga tushganda st.secrets bo'lmasligi mumkin – e'tiborsiz qoldiramiz
        pass

    # 2. Fallback – yuqoridagi POSTGRES_URL constant
    if POSTGRES_URL and isinstance(POSTGRES_URL, str) and POSTGRES_URL.strip():
        return POSTGRES_URL.strip()

    # 3. Umuman topilmasa – xato
    raise RuntimeError(
        "Postgres URL topilmadi. Iltimos:\n"
        "- .streamlit/secrets.toml ichida [postgres] va url = \"...\" ni yozing, yoki\n"
        "- app.py dagi POSTGRES_URL constantini to'ldiring."
    )


def get_connection():
    """
    Neon PostgreSQL bazasiga ulanadi.
    URL _resolve_db_url() orqali olinadi.
    """
    db_url = _resolve_db_url()
    conn = psycopg2.connect(db_url)
    return conn


def init_db():
    """Jadvallarni yaratish va default provayder akkauntini tayyorlash (PostgreSQL syntaksis)."""
    conn = get_connection()
    cur = conn.cursor()

    # Providers – PT provayderlar (adminlar)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            full_name TEXT
        );
    """)

    # Labs – PT qatnashchi laboratoriyalar
    cur.execute("""
        CREATE TABLE IF NOT EXISTS labs (
            lab_code TEXT PRIMARY KEY,
            lab_name TEXT,
            password_hash TEXT NOT NULL,
            contact_email TEXT
        );
    """)

    # Schemes – MT sxemalar
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schemes (
            scheme_code TEXT PRIMARY KEY,
            measurand TEXT,
            unit TEXT,
            assigned_value DOUBLE PRECISION,
            sigma_pt DOUBLE PRECISION,
            u_ref DOUBLE PRECISION,
            description TEXT,
            start_date TEXT,
            end_date TEXT
        );
    """)

    # Results – laboratoriya natijalari
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id SERIAL PRIMARY KEY,
            datetime_utc TEXT,
            scheme_code TEXT REFERENCES schemes(scheme_code),
            lab_code TEXT REFERENCES labs(lab_code),
            measurand TEXT,
            result_mgkg DOUBLE PRECISION,
            u_lab_mgkg DOUBLE PRECISION,
            replicate_id TEXT
        );
    """)

    # Risks – risklar reestri
    cur.execute("""
        CREATE TABLE IF NOT EXISTS risks (
            id SERIAL PRIMARY KEY,
            title TEXT,
            category TEXT,
            description TEXT,
            severity INTEGER,
            likelihood INTEGER,
            rating INTEGER,
            level TEXT,
            status TEXT,
            owner TEXT,
            created_at TEXT,
            updated_at TEXT
        );
    """)

    # Suppliers – tashqi yetkazib beruvchilar
    cur.execute("""
        CREATE TABLE IF NOT EXISTS suppliers (
            id SERIAL PRIMARY KEY,
            name TEXT,
            category TEXT,
            contact TEXT,
            status TEXT,
            score DOUBLE PRECISION,
            last_eval TEXT,
            notes TEXT
        );
    """)

    # Default provider (faqat DB ichida, UI’da ko‘rsatilmaydi)
    cur.execute("SELECT COUNT(*) FROM providers;")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO providers (username, password_hash, full_name) VALUES (%s, %s, %s);",
            ("Umidjon", hash_password("Umidjon1771"), "Default Admin")
        )

    conn.commit()
    conn.close()


# ---------- PROVIDER SERVICE ----------

def get_provider(username):
    """
    username bo‘yicha provayderni dict ko‘rinishida qaytaradi (PostgreSQL uchun).
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT username, password_hash, full_name FROM providers WHERE username = %s;",
        (username,)
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {"username": row[0], "password_hash": row[1], "full_name": row[2]}


def validate_provider_login(username, password):
    row = get_provider(username)
    if row is None:
        return False
    return check_password(password, row["password_hash"])


# ---------- LAB SERVICE ----------

def get_lab(lab_code):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT lab_code, lab_name, password_hash, contact_email FROM labs WHERE lab_code = %s;",
        (lab_code,)
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "lab_code": row[0],
        "lab_name": row[1],
        "password_hash": row[2],
        "contact_email": row[3],
    }


def validate_lab_login(lab_code, password):
    row = get_lab(lab_code)
    if row is None:
        return False
    return check_password(password, row["password_hash"])


def upsert_lab(lab_code, lab_name, password, contact_email=""):
    conn = get_connection()
    cur = conn.cursor()
    password_hash = hash_password(password)
    cur.execute("""
        INSERT INTO labs (lab_code, lab_name, password_hash, contact_email)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(lab_code) DO UPDATE SET
            lab_name = EXCLUDED.lab_name,
            password_hash = EXCLUDED.password_hash,
            contact_email = EXCLUDED.contact_email;
    """, (lab_code, lab_name, password_hash, contact_email))
    conn.commit()
    conn.close()


def get_all_labs_df():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT lab_code, lab_name, contact_email FROM labs ORDER BY lab_code;",
        conn
    )
    conn.close()
    return df


# ---------- SCHEME SERVICE ----------

def upsert_scheme(
    scheme_code,
    measurand,
    unit,
    assigned_value,
    sigma_pt,
    u_ref,
    description,
    start_date,
    end_date
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO schemes (
            scheme_code, measurand, unit,
            assigned_value, sigma_pt, u_ref,
            description, start_date, end_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(scheme_code) DO UPDATE SET
            measurand = EXCLUDED.measurand,
            unit = EXCLUDED.unit,
            assigned_value = EXCLUDED.assigned_value,
            sigma_pt = EXCLUDED.sigma_pt,
            u_ref = EXCLUDED.u_ref,
            description = EXCLUDED.description,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date;
    """, (
        scheme_code, measurand, unit,
        assigned_value, sigma_pt, u_ref,
        description, start_date, end_date
    ))
    conn.commit()
    conn.close()


def get_scheme(scheme_code):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT scheme_code, measurand, unit, assigned_value, sigma_pt, u_ref, description, start_date, end_date "
        "FROM schemes WHERE scheme_code = %s;",
        (scheme_code,)
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "scheme_code": row[0],
        "measurand": row[1],
        "unit": row[2],
        "assigned_value": row[3],
        "sigma_pt": row[4],
        "u_ref": row[5],
        "description": row[6],
        "start_date": row[7],
        "end_date": row[8],
    }


def get_all_schemes_df():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM schemes ORDER BY scheme_code;",
        conn
    )
    conn.close()
    return df


# ---------- RESULT SERVICE ----------

def insert_result(
    scheme_code,
    lab_code,
    measurand,
    result_mgkg,
    u_lab_mgkg,
    replicate_id
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO results (
            datetime_utc, scheme_code, lab_code,
            measurand, result_mgkg, u_lab_mgkg, replicate_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (
        datetime.utcnow().isoformat(),
        scheme_code,
        lab_code,
        measurand,
        result_mgkg,
        u_lab_mgkg if u_lab_mgkg is not None else None,
        replicate_id
    ))
    conn.commit()
    conn.close()


def get_results_df_for_scheme(scheme_code):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM results WHERE scheme_code = %s ORDER BY lab_code, datetime_utc;",
        conn,
        params=(scheme_code,)
    )
    conn.close()
    return df


# ---------- RISK SERVICE ----------

def insert_risk(title, category, description, severity, likelihood, owner):
    rating = severity * likelihood
    if rating >= 15:
        level = "High"
    elif rating >= 8:
        level = "Medium"
    else:
        level = "Low"

    status = "Open"
    now = datetime.utcnow().isoformat()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO risks (
            title, category, description,
            severity, likelihood, rating, level,
            status, owner, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """, (
        title, category, description,
        severity, likelihood, rating, level,
        status, owner, now, now
    ))
    conn.commit()
    conn.close()


def get_risks_df():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM risks ORDER BY rating DESC, created_at DESC;",
        conn
    )
    conn.close()
    return df


def update_risk_status(risk_id, new_status):
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE risks SET status = %s, updated_at = %s WHERE id = %s;",
        (new_status, now, risk_id)
    )
    conn.commit()
    conn.close()


# ---------- SUPPLIER SERVICE ----------

def insert_supplier(name, category, contact, status, score, last_eval, notes):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO suppliers (
            name, category, contact,
            status, score, last_eval, notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (
        name, category, contact,
        status, score, last_eval, notes
    ))
    conn.commit()
    conn.close()


def get_suppliers_df():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM suppliers ORDER BY status, score DESC;",
        conn
    )
    conn.close()
    return df


def update_supplier_status(supplier_id, new_status):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE suppliers SET status = %s WHERE id = %s;",
        (new_status, supplier_id)
    )
    conn.commit()
    conn.close()


# ============================================
# 4. STATISTIK FUNKSIYALAR (Z, En, ANOVA, REG)
# ============================================

def compute_z_scores(df, value_col, assigned_col, sigma_col, new_col="z_score"):
    df = df.copy()
    df[new_col] = (df[value_col] - df[assigned_col]) / df[sigma_col]
    return df


def compute_z_prime(df, value_col, assigned_col, sigma_pt_col, u_ref_col, new_col="z_prime"):
    df = df.copy()
    df[new_col] = (df[value_col] - df[assigned_col]) / np.sqrt(
        df[sigma_pt_col] ** 2 + df[u_ref_col] ** 2
    )
    return df


def compute_en(df, value_col, assigned_col, u_lab_col, u_ref_col, new_col="En"):
    df = df.copy()
    df[new_col] = (df[value_col] - df[assigned_col]) / np.sqrt(
        df[u_lab_col] ** 2 + df[u_ref_col] ** 2
    )
    return df


def compute_zeta(df, value_col, assigned_col, u_lab_col, u_ref_col, new_col="zeta"):
    return compute_en(df, value_col, assigned_col, u_lab_col, u_ref_col, new_col=new_col)


def classify_z(z_value):
    """Z-score interpretatsiyasi (ISO 13528 bo‘yicha oddiy qoida)."""
    if z_value is None or pd.isna(z_value):
        return ""
    z = abs(z_value)
    if z <= 2:
        return "Satisfactory"
    elif z <= 3:
        return "Questionable"
    else:
        return "Unsatisfactory"


# ---------- GOMOGENLIK (ANOVA) ----------

def homogeneity_anova(df, bottle_col="bottle", value_col="result_mgkg"):
    df = df.copy()
    df = df.dropna(subset=[bottle_col, value_col])
    overall_mean = df[value_col].mean()

    groups = df.groupby(bottle_col)
    k = groups.ngroups
    N = len(df)

    ss_between = 0.0
    ss_within = 0.0
    for _, g in groups:
        m_i = g[value_col].mean()
        n = len(g)
        ss_between += n * (m_i - overall_mean) ** 2
        ss_within += ((g[value_col] - m_i) ** 2).sum()

    df_between = k - 1
    df_within = N - k

    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    s_bb = np.sqrt(max(ms_between - ms_within, 0)) if not np.isnan(ms_between) and not np.isnan(ms_within) else np.nan
    s_w = np.sqrt(ms_within) if not np.isnan(ms_within) else np.nan

    return {
        "grand_mean": overall_mean,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "s_bb": s_bb,
        "s_w": s_w
    }


def homogeneity_judgement(s_bb, sigma_pt, factor=0.3):
    if pd.isna(s_bb) or pd.isna(sigma_pt):
        return "Hisoblash uchun ma'lumot yetarli emas"
    limit = factor * sigma_pt
    if s_bb <= limit:
        return f"Gomogenlik QONIQARLI (s_bb = {round_2(s_bb)}, limit = {round_2(limit)})"
    else:
        return f"Gomogenlik QONIQARSIZ (s_bb = {round_2(s_bb)}, limit = {round_2(limit)})"


# ---------- STABILLIK (REGRESSIYA) ----------

def stability_regression(df, time_col="time_days", value_col="result_mgkg"):
    df = df.copy().dropna(subset=[time_col, value_col])
    x = df[time_col].values.astype(float)
    y = df[value_col].values.astype(float)

    n = len(x)
    if n < 2:
        return None

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))

    b = Sxy / Sxx
    a = y_mean - b * x_mean

    y_pred = a + b * x
    residuals = y - y_pred
    s2 = np.sum(residuals ** 2) / (n - 2)
    s = np.sqrt(s2)
    se_b = np.sqrt(s2 / Sxx)

    return {
        "intercept": a,
        "slope": b,
        "sigma_res": s,
        "se_slope": se_b,
        "n": n
    }


def stability_judgement(slope, time_interval_days, sigma_pt):
    if any(pd.isna(x) for x in [slope, sigma_pt, time_interval_days]):
        return "Baholash uchun ma'lumot yetarli emas"

    delta = abs(slope) * time_interval_days
    limit = 0.3 * sigma_pt
    if delta <= limit:
        return f"Stabillik QONIQARLI (|b|*T = {round_2(delta)}, limit = {round_2(limit)})"
    else:
        return f"Stabillik QONIQARSIZ (|b|*T = {round_2(delta)}, limit = {round_2(limit)})"


# ============================================
# 5. LOGIN / ROLLAR (UI)
# ============================================

def login_sidebar():
    """Sidebar orqali rol va loginni boshqarish."""
    render_logo_sidebar()
    st.sidebar.title("Kirish")

    role = st.sidebar.radio("Rolni tanlang", ["PT qatnashchisi", "PT provayder"], index=0)

    if role == "PT provayder":
        st.sidebar.subheader("PT provayder login")
        username = st.sidebar.text_input("Login", key="prov_username")
        password = st.sidebar.text_input("Parol", type="password", key="prov_password")

        if username and password:
            if validate_provider_login(username, password):
                st.sidebar.success("Kirish muvaffaqiyatli ✅")
                return "provider", {"username": username}
            else:
                st.sidebar.error("Login yoki parol noto'g'ri ❌")
        return None, None

    else:
        st.sidebar.subheader("Laboratoriya login")
        lab_code = st.sidebar.text_input("Laboratoriya kodi (masalan, LAB-001)", key="lab_code")
        lab_pass = st.sidebar.text_input("Laboratoriya paroli", type="password", key="lab_pass")
        scheme_code = st.sidebar.text_input("MT sxema kodi (masalan, TPF-Au-2025-01)", key="scheme_code")

        if lab_code and lab_pass and scheme_code:
            if validate_lab_login(lab_code, lab_pass):
                st.sidebar.success("Laboratoriya login muvaffaqiyatli ✅")
                return "lab", {"lab_code": lab_code.strip(), "scheme_code": scheme_code.strip()}
            else:
                st.sidebar.error("Laboratoriya kodi yoki paroli noto'g'ri ❌")
        return None, None


# ============================================
# 6. LABORATORIYA UI (PT QATNASHCHISI)
# ============================================

def lab_page(lab_ctx):
    lab_code = lab_ctx["lab_code"]
    scheme_code = lab_ctx["scheme_code"]

    st.title("PT Suite – PT qatnashchisi (Laboratoriya)")

    scheme = get_scheme(scheme_code)
    if scheme is None:
        st.error("Bu sxema provayder tomonidan bazaga kiritilmagan. Provayder bilan bog‘laning.")
        return

    st.info(
        f"MT sxema: **{scheme_code}** – {scheme['measurand']} ({scheme['unit']}) | "
        f"Laboratoriya: **{lab_code}**"
    )

    st.markdown("### 1. Tahlil natijasini kiritish (birlik: mg/kg, kamida 2 ta kasr bilan)")

    measurand = scheme["measurand"] or st.text_input("Measurand (masalan, Au, Cu, Pb...)", value="Au")

    result = st.number_input(
        "Natija (mg/kg)",
        format="%.2f",
        step=0.01,
        min_value=0.0
    )

    u_lab = st.number_input(
        "Kengaytirilgan o‘lchash noaniqligi (mg/kg) – ixtiyoriy",
        format="%.2f",
        step=0.01,
        min_value=0.0,
        value=0.0
    )

    replicate_id = st.text_input("Laboratoriya ichidagi kod / replicat (ixtiyoriy)", value="")

    if st.button("Natijani jo'natish"):
        insert_result(
            scheme_code=scheme_code,
            lab_code=lab_code,
            measurand=measurand,
            result_mgkg=round(result, 2),
            u_lab_mgkg=round(u_lab, 2) if u_lab else None,
            replicate_id=replicate_id
        )
        st.success("Natijangiz muvaffaqiyatli yuborildi. Rahmat! ✅")
        st.caption("Eslatma: boshqa laboratoriyalarning natijalari va sxema sozlamalarini ko‘ra olmaysiz.")


# ============================================
# 7. PT PROVAYDER UI
# ============================================

def provider_page(provider_ctx):
    st.title("PT Suite – PT provayder paneli")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ MT sxemalar",
        "2️⃣ Laboratoriyalar",
        "3️⃣ Natijalar & statistik tahlil",
        "4️⃣ Gomogenlik & stabillik",
        "5️⃣ Hisobotlar",
        "6️⃣ Risklar & yetkazib beruvchilar"
    ])

    with tab1:
        ui_schemes_admin()
    with tab2:
        ui_labs_admin()
    with tab3:
        ui_statistics()
    with tab4:
        ui_homogeneity_stability()
    with tab5:
        ui_reports()
    with tab6:
        ui_risks_and_suppliers()


# ---------- 7.1. SXEMALAR ADMIN UI ----------

def ui_schemes_admin():
    st.subheader("1️⃣ MT sxemalar va ularning sozlamalari")

    st.markdown("#### Mavjud sxemalar ro‘yxati")
    df_schemes = get_all_schemes_df()
    if df_schemes.empty:
        st.info("Hozircha sxemalar bazaga kiritilmagan.")
    else:
        st.dataframe(df_schemes)

    st.markdown("#### Yangi sxema qo‘shish yoki mavjudini yangilash")

    with st.form("scheme_form"):
        col1, col2 = st.columns(2)
        with col1:
            scheme_code = st.text_input("Sxema kodi", value="TPF-Au-2025-01")
            measurand = st.text_input("Measurand (masalan, Au)", value="Au")
            unit = st.text_input("Birlik", value="mg/kg")
            assigned_value = st.number_input("Assigned value Xref (mg/kg)", step=0.01, format="%.4f")
        with col2:
            sigma_pt = st.number_input("Sigma_pt (mg/kg)", step=0.0001, format="%.4f")
            u_ref = st.number_input("Xref noaniqligi u_ref (mg/kg)", step=0.0001, format="%.4f")
            start_date = st.text_input("Boshlanish sanasi (ixtiyoriy)", value="")
            end_date = st.text_input("Tugash sanasi (ixtiyoriy)", value="")

        description = st.text_area("Sxema haqida qisqacha izoh", value="", height=80)

        submitted = st.form_submit_button("Saqlash / yangilash")
        if submitted:
            if not scheme_code:
                st.error("Sxema kodi majburiy.")
            else:
                upsert_scheme(
                    scheme_code=scheme_code.strip(),
                    measurand=measurand.strip(),
                    unit=unit.strip(),
                    assigned_value=float(assigned_value) if assigned_value else None,
                    sigma_pt=float(sigma_pt) if sigma_pt else None,
                    u_ref=float(u_ref) if u_ref else None,
                    description=description.strip(),
                    start_date=start_date.strip(),
                    end_date=end_date.strip()
                )
                st.success(f"{scheme_code} sxemasi saqlandi / yangilandi.")


# ---------- 7.2. LABORATORIYALAR ADMIN UI ----------

def ui_labs_admin():
    st.subheader("2️⃣ Laboratoriyalar va ularning akkauntlari")

    st.markdown("#### Mavjud laboratoriyalar ro‘yxati")
    df_labs = get_all_labs_df()
    if df_labs.empty:
        st.info("Hozircha laboratoriyalar bazaga kiritilmagan.")
    else:
        st.dataframe(df_labs)

    st.markdown("#### Yangi laboratoriya qo‘shish / parolini yangilash")

    with st.form("lab_form"):
        col1, col2 = st.columns(2)
        with col1:
            lab_code = st.text_input("Laboratoriya kodi (masalan, LAB-001)", "")
            lab_name = st.text_input("Laboratoriya nomi", "")
        with col2:
            lab_pass = st.text_input("Yangi parol", type="password")
            contact_email = st.text_input("Kontakt email", "")

        submitted = st.form_submit_button("Saqlash / yangilash")
        if submitted:
            if not (lab_code and lab_pass):
                st.error("Laboratoriya kodi va parol majburiy.")
            else:
                upsert_lab(
                    lab_code=lab_code.strip(),
                    lab_name=lab_name.strip(),
                    password=lab_pass.strip(),
                    contact_email=contact_email.strip()
                )
                st.success(f"{lab_code} laboratoriyasi saqlandi / yangilandi.")


# ---------- 7.3. STATISTIK TAHLIL UI ----------

def ui_statistics():
    st.subheader("3️⃣ Natijalar va statistik tahlil (Z, Z', En, zeta)")

    df_schemes = get_all_schemes_df()
    if df_schemes.empty:
        st.warning("Avval kamida bitta sxema yarating.")
        return

    scheme_code = st.selectbox(
        "MT sxema kodi",
        options=df_schemes["scheme_code"].tolist()
    )

    scheme = get_scheme(scheme_code)
    if scheme is None:
        st.error("Sxema topilmadi.")
        return

    df = get_results_df_for_scheme(scheme_code)
    if df.empty:
        st.info("Bu sxema bo‘yicha natijalar hali yo‘q.")
        return

    st.markdown("#### Xom natijalar (mg/kg)")
    st.dataframe(df)

    st.info(
        f"Sxema sozlamalari: Xref = {round_2(scheme['assigned_value'])} {scheme['unit']}, "
        f"σ_pt = {round_2(scheme['sigma_pt'])}, u_ref = {round_2(scheme['u_ref'])}"
    )

    df_stats = df.copy()
    df_stats["assigned_value"] = scheme["assigned_value"]
    df_stats["sigma_pt"] = scheme["sigma_pt"]
    df_stats["u_ref"] = scheme["u_ref"]

    if "u_lab_mgkg" not in df_stats.columns:
        df_stats["u_lab_mgkg"] = np.nan

    st.markdown("#### Qaysi statistikalarni hisoblaymiz?")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        calc_z = st.checkbox("Z-score", value=True)
    with col2:
        calc_zp = st.checkbox("Z'-score", value=True)
    with col3:
        calc_en = st.checkbox("En factor", value=True)
    with col4:
        calc_zeta = st.checkbox("Zeta-score", value=False)

    if st.button("Statistikani hisoblash"):
        if calc_z:
            df_stats = compute_z_scores(df_stats, "result_mgkg", "assigned_value", "sigma_pt", "z_score")

        if calc_zp and scheme["u_ref"] is not None:
            df_stats = compute_z_prime(df_stats, "result_mgkg", "assigned_value", "sigma_pt", "u_ref", "z_prime")

        if calc_en:
            df_stats = compute_en(df_stats, "result_mgkg", "assigned_value", "u_lab_mgkg", "u_ref", "En")

        if calc_zeta:
            df_stats = compute_zeta(df_stats, "result_mgkg", "assigned_value", "u_lab_mgkg", "u_ref", "zeta")

        # Z-score interpretatsiyasi uchun ustun
        if "z_score" in df_stats.columns:
            df_stats["z_verdict"] = df_stats["z_score"].apply(classify_z)

        # Raqamlarni formatlash
        numeric_cols = df_stats.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            df_stats[c] = df_stats[c].apply(
                lambda x: float(f"{x:.4f}") if not pd.isna(x) else np.nan
            )

        st.markdown("#### Hisoblangan natijalar")
        st.dataframe(df_stats)

        excel_bytes = dataframe_to_excel_bytes(df_stats, sheet_name="Statistics")
        st.download_button(
            label="📥 Statistik hisobotni Excel ko‘rinishida yuklab olish",
            data=excel_bytes,
            file_name=f"{scheme_code}_stats.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ---------- 7.4. GOMOGENLIK & STABILLIK UI ----------

def ui_homogeneity_stability():
    st.subheader("4️⃣ Gomogenlik va stabillik tahlili")

    tab_h, tab_s = st.tabs(["Gomogenlik", "Stabillik"])

    with tab_h:
        ui_homogeneity_section()

    with tab_s:
        ui_stability_section()


def ui_homogeneity_section():
    st.markdown("### Gomogenlik sinovi (bir faktorli ANOVA)")

    st.markdown(
        "Fayl strukturasiga misol:\n"
        "- `bottle` – shisha / qadoq raqami\n"
        "- `result_mgkg` – replicat natijasi (mg/kg)\n"
    )

    file = st.file_uploader("Gomogenlik ma'lumotlarini yuklang (CSV/Excel)", type=["csv", "xlsx"], key="homog_file")

    sigma_pt = st.number_input(
        "MT sxemasi uchun rejalashtirilgan sigma_pt (mg/kg)",
        format="%.4f",
        step=0.0001,
        min_value=0.0
    )

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.markdown("#### Kiritilgan ma'lumotlar")
        st.dataframe(df)

        bottle_col = st.selectbox(
            "Shisha (bottle) ustuni",
            options=df.columns,
            index=(list(df.columns).index("bottle") if "bottle" in df.columns else 0)
        )

        value_col = st.selectbox(
            "Natija ustuni (mg/kg)",
            options=df.columns,
            index=(list(df.columns).index("result_mgkg") if "result_mgkg" in df.columns else 0)
        )

        if st.button("Gomogenlikni hisoblash"):
            res = homogeneity_anova(df, bottle_col=bottle_col, value_col=value_col)

            col1, col2, col3 = st.columns(3)
            col1.metric("Umumiy o‘rtacha", round_2(res["grand_mean"]))
            col2.metric("s_bb (between-bottle SD)", round_2(res["s_bb"]))
            col3.metric("s_w (within-bottle SD)", round_2(res["s_w"]))

            st.json(res)

            if sigma_pt > 0:
                decision = homogeneity_judgement(res["s_bb"], sigma_pt)
                st.success(decision)
            else:
                st.info("sigma_pt kiritilmagan, qiyosiy baho berilmadi.")


def ui_stability_section():
    st.markdown("### Stabillik sinovi (chiziqli regressiya)")

    st.markdown(
        "Fayl strukturasiga misol:\n"
        "- `time_days` – vaqt (kunlarda, masalan 0, 7, 14, 28 ...)\n"
        "- `result_mgkg` – natija (mg/kg)\n"
    )

    file = st.file_uploader("Stabillik ma'lumotlarini yuklang (CSV/Excel)", type=["csv", "xlsx"], key="stab_file")

    sigma_pt = st.number_input(
        "MT sxemasi uchun sigma_pt (mg/kg)",
        format="%.4f",
        step=0.0001,
        min_value=0.0,
        key="stab_sigma_pt"
    )

    time_interval = st.number_input(
        "Baholanadigan maksimal vaqt oralig‘i T (kunlarda, masalan, 60 yoki 90)",
        format="%.1f",
        step=1.0,
        min_value=0.0,
        key="stab_T"
    )

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.markdown("#### Kiritilgan ma'lumotlar")
        st.dataframe(df)

        time_col = st.selectbox(
            "Vaqt ustuni (kun)",
            options=df.columns,
            index=(list(df.columns).index("time_days") if "time_days" in df.columns else 0)
        )

        value_col = st.selectbox(
            "Natija ustuni (mg/kg)",
            options=df.columns,
            index=(list(df.columns).index("result_mgkg") if "result_mgkg" in df.columns else 0)
        )

        if st.button("Stabillikni hisoblash"):
            res = stability_regression(df, time_col=time_col, value_col=value_col)
            if res is None:
                st.error("Hisoblash uchun kamida 2 ta nuqta kerak.")
                return

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Intercept (a)", round_2(res["intercept"]))
            col2.metric("Slope (b)", round_2(res["slope"]))
            col3.metric("σ_res (residual SD)", round_2(res["sigma_res"]))
            col4.metric("SE(b)", round_2(res["se_slope"]))

            st.json(res)

            if sigma_pt > 0 and time_interval > 0:
                decision = stability_judgement(res["slope"], time_interval, sigma_pt)
                st.success(decision)
            else:
                st.info("sigma_pt va T kiritilmagan, faqat regressiya parametrlari ko‘rsatildi.")


# ---------- 7.5. HISOBOTLAR UI ----------

def ui_reports():
    st.subheader("5️⃣ Hisobotlar (xom natijalar + HTML preview)")

    df_schemes = get_all_schemes_df()
    if df_schemes.empty:
        st.warning("Avval sxema yarating.")
        return

    scheme_code = st.selectbox(
        "MT sxema kodi",
        options=df_schemes["scheme_code"].tolist(),
        key="report_scheme"
    )

    df = get_results_df_for_scheme(scheme_code)
    if df.empty:
        st.info("Bu sxema bo‘yicha natijalar hali yo‘q.")
        return

    st.markdown("#### Xom natijalar (qatnashchilar topshirgan)")
    st.dataframe(df)

    excel_bytes = dataframe_to_excel_bytes(df, sheet_name="RawResults")
    st.download_button(
        label="📥 Xom natijalarni Excel ko‘rinishida yuklab olish",
        data=excel_bytes,
        file_name=f"{scheme_code}_raw_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("#### Sodda HTML hisobot (preview)")
    unique_labs = df["lab_code"].nunique() if "lab_code" in df.columns else "N/A"
    measurand = df["measurand"].iloc[0] if "measurand" in df.columns and not df.empty else "N/A"

    html_report = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>PT Suite report – {scheme_code}</title>
    </head>
    <body>
        <h2>PT Suite – PT hisobot</h2>
        <h3>Sxema: {scheme_code}</h3>
        <p><b>Measurand:</b> {measurand}</p>
        <p><b>Qatnashchilar soni:</b> {unique_labs}</p>
        <p><b>Topshirilgan natijalar soni:</b> {len(df)}</p>
        <p><b>Birligi:</b> mg/kg (kamida 2 ta kasr bilan kiritilgan).</p>
    </body>
    </html>
    """

    st.components.v1.html(html_report, height=260, scrolling=True)

    st.download_button(
        label="📥 HTML hisobotni yuklab olish",
        data=html_report.encode("utf-8"),
        file_name=f"{scheme_code}_report.html",
        mime="text/html"
    )


# ---------- 7.6. RISK & SUPPLIER UI ----------

def ui_risks_and_suppliers():
    st.subheader("6️⃣ Risklar & tashqi yetkazib beruvchilar")

    tab_risk, tab_sup = st.tabs(["Risklar reestri", "Yetkazib beruvchilar reestri"])

    with tab_risk:
        ui_risks_section()
    with tab_sup:
        ui_suppliers_section()


def ui_risks_section():
    st.markdown("### Risklar reestri (ISO 17043 / ISO 17025 risk-based thinking)")

    df_risks = get_risks_df()
    if df_risks.empty:
        st.info("Hozircha risklar bazaga kiritilmagan.")
    else:
        st.markdown("#### Mavjud risklar")
        st.dataframe(df_risks)

    st.markdown("#### Yangi risk qo‘shish")

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Risk nomi", value="MT sxemasi kechikib yuborilishi")
            category = st.selectbox(
                "Kategoriya",
                ["Operational", "Technical", "Financial", "IT / Security", "Impartiality", "Boshqa"],
                index=0
            )
            severity = st.slider("Severity (1–5)", min_value=1, max_value=5, value=3)
        with col2:
            likelihood = st.slider("Likelihood (1–5)", min_value=1, max_value=5, value=2)
            owner = st.text_input("Mas’ul shaxs / bo‘lim", value="PT manager")

        description = st.text_area(
            "Risk tavsifi",
            value="Masalan, namunalar ishtirokchi laboratoriyalarga belgilangan muddatdan kech yuborilishi natijasida natijalar to‘plami kechikishi."
        )

        submitted = st.form_submit_button("Riskni reestrga qo‘shish")
        if submitted:
            insert_risk(
                title=title.strip(),
                category=category.strip(),
                description=description.strip(),
                severity=int(severity),
                likelihood=int(likelihood),
                owner=owner.strip()
            )
            st.success("Risk muvaffaqiyatli qo‘shildi.")

    st.markdown("#### Risk statusini yangilash")
    df_risks = get_risks_df()
    if not df_risks.empty:
        risk_options = {
            f"[{row['id']}] {row['title']} ({row['status']})": int(row["id"])
            for _, row in df_risks.iterrows()
        }
        selected_label = st.selectbox("Riskni tanlang", list(risk_options.keys()))
        new_status = st.selectbox("Yangi status", ["Open", "In progress", "Closed", "Accepted"], index=0)
        if st.button("Statusni yangilash"):
            update_risk_status(risk_options[selected_label], new_status)
            st.success("Risk statusi yangilandi.")


def ui_suppliers_section():
    st.markdown("### Tashqi yetkazib beruvchilar reestri (ISO 17043 6.4, 6.7)")

    df_sup = get_suppliers_df()
    if df_sup.empty:
        st.info("Hozircha yetkazib beruvchilar bazaga kiritilmagan.")
    else:
        st.markdown("#### Mavjud yetkazib beruvchilar")
        st.dataframe(df_sup)

    st.markdown("#### Yangi yetkazib beruvchi qo‘shish")

    with st.form("supplier_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Yetkazib beruvchi nomi", value="CRM Provider X")
            category = st.selectbox(
                "Kategoriya",
                ["CRM / Reference material", "Transport / Logistics",
                 "Calibration / Maintenance", "IT xizmatlar", "Boshqa"],
                index=0
            )
            contact = st.text_input("Kontakt (email / tel / shartnoma raqami)", value="")
        with col2:
            status = st.selectbox("Status", ["Approved", "Conditional", "Blocked"], index=0)
            score = st.slider("Baholash balli (0–100)", min_value=0, max_value=100, value=85)
            last_eval = st.text_input(
                "Oxirgi baholash sanasi (ixtiyoriy)",
                value=datetime.utcnow().date().isoformat()
            )

        notes = st.text_area("Izohlar", value="Masalan, ISO 17034 akkreditatsiyalangan CRM ishlab chiqaruvchi.")

        submitted = st.form_submit_button("Yetkazib beruvchini reestrga qo‘shish")
        if submitted:
            insert_supplier(
                name=name.strip(),
                category=category.strip(),
                contact=contact.strip(),
                status=status.strip(),
                score=float(score),
                last_eval=last_eval.strip(),
                notes=notes.strip()
            )
            st.success("Yetkazib beruvchi muvaffaqiyatli qo‘shildi.")

    st.markdown("#### Yetkazib beruvchi statusini yangilash")
    df_sup = get_suppliers_df()
    if not df_sup.empty:
        sup_options = {
            f"[{row['id']}] {row['name']} ({row['status']})": int(row["id"])
            for _, row in df_sup.iterrows()
        }
        selected_label = st.selectbox("Yetkazib beruvchini tanlang", list(sup_options.keys()))
        new_status = st.selectbox("Yangi status", ["Approved", "Conditional", "Blocked"], index=0)
        if st.button("Yetkazib beruvchi statusini yangilash"):
            update_supplier_status(sup_options[selected_label], new_status)
            st.success("Yetkazib beruvchi statusi yangilandi.")


# ============================================
# 8. MAIN
# ============================================

def main():
    # DB ni ishga tushirishga harakat qilamiz
    try:
        init_db()
    except Exception as e:
        st.error(f"DB bilan ulanishda xato: {e}")
        st.stop()

    role, ctx = login_sidebar()

    if role == "lab":
        lab_page(ctx)
    elif role == "provider":
        provider_page(ctx)
    else:
        st.title("PT Suite – Proficiency Testing Platform")
        st.write(
            """
            Chap tomondagi menyudan rolni tanlang:

            - **PT qatnashchisi (Laboratoriya)** – laboratoriya kodi va paroli bilan kirib,
              faqat o‘zining natijasini kiritadi (mg/kg, kamida 2 ta kasr bilan).
            - **PT provayder** – sxemalar, laboratoriya akkauntlari, statistik tahlil,
              gomogenlik & stabillik, risklar reestri va tashqi yetkazib beruvchilarni
              boshqarish imkoniyatiga ega.
            """
        )


if __name__ == "__main__":
    main()
