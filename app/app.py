
"""Insurance Claim Fraud Detector
Run: streamlit run app/app.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.predict import load_assets, predict_claim


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ClaimWise | Insurance Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# THEME
# --------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

with st.sidebar:
    st.markdown("## 🌿 ClaimWise")
    st.caption("INSURANCE INTELLIGENCE")

    st.markdown("---")
    st.markdown("### Appearance")

    theme = st.radio(
        "Choose theme",
        ["Light", "Dark"],
        horizontal=True,
        key="theme"
    )

    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("🏠  Claim Analysis")
    st.markdown("📊  Model Overview")
    st.markdown("🛡️  About the Project")

    st.markdown("---")
    st.caption("Insurance Fraud Detection")
    st.caption("Machine Learning Project")


if theme == "Light":
    bg = "#F3F0E7"
    card = "#FFFCF5"
    text = "#30382B"
    muted = "#777D6D"
    olive = "#69794D"
    border = "#E2DDCF"
    input_bg = "#FFFFFF"
    sidebar = "#EAE6D9"
else:
    bg = "#18221B"
    card = "#263329"
    text = "#F0F1E9"
    muted = "#B2BBAA"
    olive = "#A5B785"
    border = "#3D4C3F"
    input_bg = "#303D32"
    sidebar = "#202D23"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    .stApp {{
        background-color: {bg};
        color: {text};
        font-family: 'DM Sans', sans-serif;
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar};
        border-right: 1px solid {border};
    }}

    [data-testid="stHeader"] {{
        background-color: {bg};
    }}

    h1, h2, h3, p, label {{
        color: {text};
        font-family: 'DM Sans', sans-serif;
    }}

    h1 {{
        font-weight: 800 !important;
        letter-spacing: -1px;
    }}

    [data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        padding: 20px;
        border-radius: 14px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {muted} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {text} !important;
    }}

    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: {input_bg};
        border-radius: 9px;
    }}

    div.stButton > button {{
        background-color: {olive};
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        width: 100%;
        min-height: 48px;
        transition: 0.2s ease;
    }}

    div.stButton > button:hover {{
        filter: brightness(1.12);
        border: none;
        color: white;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {card};
        border-radius: 14px;
        border-color: {border};
    }}

    hr {{
        border-color: {border};
    }}

    .hero {{
        background: {card};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 25px;
    }}

    .eyebrow {{
        color: {olive};
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
    }}

    .hero-title {{
        color: {text};
        font-size: 34px;
        font-weight: 800;
        margin-top: 8px;
    }}

    .hero-subtitle {{
        color: {muted};
        font-size: 15px;
        margin-top: 8px;
    }}

    .section-label {{
        color: {olive};
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }}

    .result-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
    }}

    footer {{
        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def get_assets():
    return load_assets()


try:
    model, assets, model_name = get_assets()

except FileNotFoundError:
    st.error(
        "Model files not found. Run python run_all.py first, "
        "then restart the app."
    )
    st.stop()


opts = assets["options"]


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">INSURANCE INTELLIGENCE PLATFORM</div>
        <div class="hero-title">Claim Analysis Dashboard</div>
        <div class="hero-subtitle">
            Intelligent claim assessment powered by machine learning.
            Identify potential fraud patterns and support investigation decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

m1, m2 = st.columns(2)

with m1:
    st.metric("Active ML Model", model_name)

with m2:
    st.metric("Analysis Type", "Fraud Risk Prediction")


st.markdown("---")


# --------------------------------------------------
# CLAIM INPUT FORM
# --------------------------------------------------

st.markdown("## Claim Information")
st.caption("Enter the details below to evaluate an insurance claim.")


with st.form("claim_form"):

    # INCIDENT DETAILS
    st.markdown(
        '<div class="section-label">01 / INCIDENT DETAILS</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        severity = st.selectbox(
            "Incident severity",
            opts["incident_severity"]
        )

        incident_type = st.selectbox(
            "Incident type",
            opts["incident_type"]
        )

        collision = st.selectbox(
            "Collision type",
            ["Missing"] + opts["collision_type"]
        )

        authorities = st.selectbox(
            "Authorities contacted",
            opts["authorities_contacted"]
        )

    with c2:
        police = st.selectbox(
            "Police report available",
            ["Missing"] + opts["police_report_available"]
        )

        damage = st.selectbox(
            "Property damage",
            ["Missing"] + opts["property_damage"]
        )

        hobby = st.selectbox(
            "Insured hobby",
            opts["insured_hobbies"]
        )

        witnesses = st.number_input(
            "Number of witnesses",
            0, 10, 1
        )

    st.markdown("---")

    # CUSTOMER DETAILS
    st.markdown(
        '<div class="section-label">02 / CUSTOMER & POLICY DETAILS</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age of insured",
            18, 90, 38
        )

        months = st.number_input(
            "Months as customer",
            0, 600, 200
        )

        bind = st.date_input(
            "Policy start date",
            date(2010, 1, 1),
            min_value=date(1980, 1, 1)
        )

    with c2:
        premium = st.number_input(
            "Annual premium",
            100.0, 5000.0, 1250.0
        )

        incident = st.date_input(
            "Incident date",
            date(2015, 2, 1),
            min_value=date(1980, 1, 1)
        )

    st.markdown("---")

    # CLAIM AMOUNTS
    st.markdown(
        '<div class="section-label">03 / CLAIM AMOUNTS</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        injury = st.number_input(
            "Injury claim amount",
            0, 200000, 5000
        )

    with c2:
        prop = st.number_input(
            "Property claim amount",
            0, 200000, 8000
        )

    with c3:
        vehicle = st.number_input(
            "Vehicle claim amount",
            0, 400000, 30000
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "Analyze Insurance Claim"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    if incident < bind:
        st.error(
            "Incident date cannot be earlier than policy start date."
        )
        st.stop()

    inputs = {
        "incident_severity": severity,
        "incident_type": incident_type,
        "collision_type": collision,
        "authorities_contacted": authorities,
        "police_report_available": police,
        "property_damage": damage,
        "insured_hobbies": hobby,
        "age": age,
        "months_as_customer": months,
        "policy_annual_premium": premium,
        "injury_claim": injury,
        "property_claim": prop,
        "vehicle_claim": vehicle,
        "witnesses": witnesses,
        "policy_bind_date": bind.isoformat(),
        "incident_date": incident.isoformat(),
    }

    proba, label = predict_claim(model, assets, inputs)

    st.markdown("---")
    st.markdown("## Claim Assessment Results")

    st.progress(
        min(max(float(proba), 0.0), 1.0)
    )

    st.metric(
        "Predicted fraud risk score",
        f"{proba:.1%}"
    )

    if label == 1:

        st.error(
            "⚠️ POSSIBLE FRAUD DETECTED\n\n"
            "This claim has been flagged for further manual investigation."
        )

    else:

        st.success(
            "✓ NO FRAUD FLAG\n\n"
            "The model did not flag this claim as potentially fraudulent."
        )

    st.caption(
        "This is a machine learning prediction, not a final "
        "insurance or fraud determination. All flagged claims "
        "require human review."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:15px;">
        <p style="font-size:13px;">
            ClaimWise · Insurance Fraud Detection System
        </p>
        <p style="font-size:11px;">
            Powered by Machine Learning | Academic Project
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
