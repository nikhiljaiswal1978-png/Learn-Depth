import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Sentinel AI | Transaction Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM UI STYLING
# UI ONLY — MODEL/PREDICTION LOGIC IS UNCHANGED
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(59,130,246,0.14), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(139,92,246,0.12), transparent 25%),
            #07111f;
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide default Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero */
    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(30,41,59,.88));
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 20px 50px rgba(0,0,0,.25);
        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(59,130,246,.14);
        border: 1px solid rgba(96,165,250,.25);
        color: #93c5fd;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .4px;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: 42px;
        line-height: 1.1;
        margin: 0;
        color: #ffffff;
        letter-spacing: -1px;
    }

    .hero p {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 12px;
        margin-bottom: 0;
    }

    /* Section titles */
    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #f8fafc;
        margin: 8px 0 4px 0;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 16px;
    }

    /* Cards */
    .glass-card {
        padding: 22px;
        border-radius: 20px;
        background: rgba(15,23,42,.76);
        border: 1px solid rgba(148,163,184,.16);
        box-shadow: 0 14px 35px rgba(0,0,0,.18);
        min-height: 110px;
    }

    .card-label {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .card-value {
        color: #f8fafc;
        font-size: 27px;
        font-weight: 800;
    }

    /* Input area */
    div[data-testid="stForm"] {
        background: rgba(15,23,42,.72);
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 14px 35px rgba(0,0,0,.18);
    }

    /* Labels */
    label, .stSlider label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #111c2e !important;
        border-color: #334155 !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    }
            
    /* Button */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        padding: 12px 18px;
        font-weight: 800;
        font-size: 15px;
        color: white;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        box-shadow: 0 10px 25px rgba(37,99,235,.28);
        transition: all .2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(124,58,237,.30);
    }

    /* Result cards */
    .result-safe {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(6,78,59,.42), rgba(15,118,110,.20));
        border: 1px solid rgba(52,211,153,.30);
        text-align: center;
        box-shadow: 0 18px 45px rgba(0,0,0,.22);
    }

    .result-risk {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(127,29,29,.45), rgba(153,27,27,.18));
        border: 1px solid rgba(248,113,113,.35);
        text-align: center;
        box-shadow: 0 18px 45px rgba(0,0,0,.22);
    }

    .result-icon {
        font-size: 46px;
        margin-bottom: 8px;
    }

    .result-title {
        font-size: 27px;
        font-weight: 850;
        color: #ffffff;
        margin-bottom: 7px;
    }

    .result-text {
        color: #cbd5e1;
        font-size: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 34px;
        padding-top: 20px;
        border-top: 1px solid rgba(148,163,184,.10);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #091525;
        border-right: 1px solid rgba(148,163,184,.12);
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 3px;
    }

    .sidebar-text {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Mobile */
    @media (max-width: 768px) {
        .hero h1 {
            font-size: 30px;
        }

        .hero {
            padding: 22px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD THE SAME MODEL
# ============================================================
from pathlib import Path

# Project root = Payment-Transaction-Anomaly-Classification
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "anomaly_model.pkl"

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
FINAL_MODEL_NAME = artifact.get("final_model", "Logistic Regression")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛡️ Sentinel AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">'
        'Payment transaction anomaly classification using a machine-learning model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-title">⚙️ Model</div>', unsafe_allow_html=True)
    st.info(f"{FINAL_MODEL_NAME}")

    st.markdown('<div class="sidebar-title">📌 Inputs</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-text">
        • Transaction amount<br>
        • Transaction hour<br>
        • 24h transaction frequency<br>
        • Merchant category risk<br>
        • Account age<br>
        • 30-day average amount
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("Educational prototype • Not a production fraud-detection system")

# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">FINTECH • MACHINE LEARNING • SECURITY</div>
    <h1>🛡️ Payment Transaction<br>Anomaly Detection</h1>
    <p>
        Analyze transaction behaviour and identify potentially anomalous
        payment activity using a trained machine-learning model.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP INFO CARDS
# ============================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">SYSTEM</div>
        <div class="card-value">Sentinel AI</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-label">MODEL</div>
        <div class="card-value">{FINAL_MODEL_NAME}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">OUTPUT</div>
        <div class="card-value">Routine / Anomaly</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="section-title">🔍 Transaction Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Enter the transaction characteristics below and run the ML analysis.</div>',
    unsafe_allow_html=True
)

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "💰 Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        hour = st.slider(
            "🕐 Transaction Hour",
            0,
            23,
            12
        )

        freq = st.slider(
            "🔄 Transaction Frequency — Last 24h",
            0,
            20,
            3
        )

    with col2:
        risk = st.slider(
            "🏪 Merchant Category Risk",
            0,
            5,
            2
        )

        age = st.number_input(
            "📅 Account Age — Days",
            min_value=0,
            value=365,
            step=1
        )

        avg = st.number_input(
            "📊 Average Transaction Amount — 30d",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 ANALYZE TRANSACTION")

# ============================================================
# PREDICTION
# SAME PREDICTION LOGIC — ONLY PRESENTATION CHANGED
# ============================================================
if submitted:
    row = pd.DataFrame(
        [[amount, hour, freq, risk, age, avg]],
        columns=features
    )

    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📡 Analysis Result</div>', unsafe_allow_html=True)

    result_col, details_col = st.columns([1.35, 1])

    with result_col:
        if pred == 1:
            st.markdown(f"""
            <div class="result-risk">
                <div class="result-icon">🚨</div>
                <div class="result-title">Anomalous Transaction</div>
                <div class="result-text">
                    The model classified this transaction as anomalous / fraud-like.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">Routine Transaction</div>
                <div class="result-text">
                    The model classified this transaction as routine.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with details_col:
        st.markdown("""
        <div class="glass-card">
            <div class="card-label">PREDICTED ANOMALY PROBABILITY</div>
        """, unsafe_allow_html=True)

        st.metric(
            "Model Probability",
            f"{prob:.2%}"
        )

        st.progress(min(max(prob, 0.0), 1.0))

        if prob < 0.25:
            risk_label = "LOW"
        elif prob < 0.60:
            risk_label = "MODERATE"
        else:
            risk_label = "HIGH"

        st.markdown(
            f'<div class="card-label">MODEL OUTPUT LEVEL: <b>{risk_label}</b></div>',
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "ℹ️ The probability shown above is the model's output. "
        "It should not be interpreted as a real-world financial risk score."
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    🛡️ Sentinel AI &nbsp;•&nbsp; Payment Transaction Anomaly Classification<br>
    Educational Machine Learning Prototype &nbsp;•&nbsp; Built with Python + Streamlit
</div>
""", unsafe_allow_html=True)
