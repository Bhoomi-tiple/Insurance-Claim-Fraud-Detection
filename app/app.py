"""ClaimWise AI - Insurance Claim & Fraud Intelligence Portal
Run:  streamlit run app/app.py

Tabs
  1. Claim Portal       : Real-world 3-step FNOL claim intake + live Claim Summary Receipt +
                          Adjuster Triage Dossier (Red Flags, What-If Simulator, Sensitivity).
  2. Batch Triage       : Upload a CSV of claims -> prioritized SIU/Desk/Fast-Track worklist.
  3. Model & Validation : Held-out test metrics, triage policy, figures, CV/ablation tables.
"""
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FIGURES, PROCESSED_DIR, RESULTS, TABLES, TARGET
from src.data import ENGINEERED_COLS
from src.explain_local import FEATURE_LABELS, explain_claim, fmt_value
from src.predict import (CLAIM_PARTS, build_row, coerce_types, is_missing, load_assets,
                         load_background, load_triage, prepare_batch)
from src.triage import TIER_ORDER, TIERS, risk_scores, tier_of

st.set_page_config(
    page_title="ClaimWise AI | Insurance Claims & Fraud Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ss = st.session_state

# --------------------------------------------------
# ENTERPRISE INSURTECH PALETTE (WCAG AA/AAA Contrast)
# --------------------------------------------------
BG = "#F1F5F9"
CARD = "#FFFFFF"
SUBCARD = "#F8FAFC"
SIDEBAR_BG = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#475569"
BORDER = "#CBD5E1"
BORDER_SOFT = "#E2E8F0"
PRIMARY = "#1D4ED8"
PRIMARY_HOVER = "#1E40AF"
PRIMARY_SOFT = "#EFF6FF"

TIER_STYLES = {
    "HIGH":   {"accent": "#DC2626", "bg": "#FEF2F2", "fg": "#991B1B", "border": "#FECACA", "status": "SIU INVESTIGATION HOLD"},
    "MEDIUM": {"accent": "#D97706", "bg": "#FFFBEB", "fg": "#92400E", "border": "#FDE68A", "status": "DESK REVIEW REQUIRED"},
    "LOW":    {"accent": "#059669", "bg": "#ECFDF5", "fg": "#065F46", "border": "#A7F3D0", "status": "FAST-TRACK ELIGIBLE"},
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
[data-testid="stHeader"] {{ background-color: {BG} !important; }}
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER_SOFT} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT}; }}

h1, h2, h3, h4, h5, h6, p, li, label, span {{
    color: {TEXT};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
h1, h2, h3 {{ font-weight: 700 !important; letter-spacing: -0.02em; }}

[data-testid="stCaptionContainer"], .stCaption, small {{
    color: {MUTED} !important;
    font-size: 0.875rem !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER_SOFT} !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}
[data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 800 !important; }}

/* Inputs & Dropdowns */
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextInput"] input {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
}}
[data-baseweb="menu"] li {{ color: {TEXT} !important; }}
[data-baseweb="menu"] li:hover {{ background-color: {PRIMARY_SOFT} !important; }}

/* Buttons */
div.stButton > button {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 9px 14px !important;
    font-weight: 600 !important;
    width: 100%;
    min-height: 42px;
    transition: all 0.15s ease;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}
div.stButton > button:hover {{
    border-color: {PRIMARY} !important;
    color: {PRIMARY} !important;
    background-color: {PRIMARY_SOFT} !important;
}}
div.stFormSubmitButton > button,
div.stDownloadButton > button {{
    background-color: {PRIMARY} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-weight: 700 !important;
    width: 100%;
    min-height: 48px;
    box-shadow: 0 2px 6px rgba(29, 78, 216, 0.25);
}}
div.stFormSubmitButton > button p,
div.stDownloadButton > button p {{ color: #FFFFFF !important; }}
div.stFormSubmitButton > button:hover,
div.stDownloadButton > button:hover {{
    background-color: {PRIMARY_HOVER} !important;
    color: #FFFFFF !important;
}}

/* Containers */
[data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {CARD} !important;
    border: 1px solid {BORDER_SOFT} !important;
    border-radius: 14px !important;
    padding: 8px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 8px;
    border-bottom: 2px solid {BORDER_SOFT};
    padding-bottom: 4px;
}}
[data-testid="stTabs"] button[role="tab"] {{
    background-color: transparent !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 18px !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background-color: {CARD} !important;
    color: {PRIMARY} !important;
    border-bottom: 3px solid {PRIMARY} !important;
    font-weight: 700 !important;
}}
[data-testid="stTabs"] button[role="tab"] p {{ color: inherit !important; }}

/* Expander */
[data-testid="stExpander"] {{
    background-color: {SUBCARD} !important;
    border: 1px solid {BORDER_SOFT} !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{
    color: {TEXT} !important;
    font-weight: 600 !important;
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background-color: {SUBCARD} !important;
    border: 1.5px dashed {BORDER} !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploaderDropzone"] * {{ color: {TEXT} !important; }}

hr {{ border-color: {BORDER_SOFT} !important; margin: 1rem 0 !important; }}

/* Portal Header */
.portal-nav {{
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}}
.portal-eyebrow {{
    color: #93C5FD !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
}}
.portal-title {{
    color: #FFFFFF !important;
    font-size: 26px;
    font-weight: 800;
    margin-top: 4px;
    letter-spacing: -0.02em;
}}
.portal-sub {{
    color: #CBD5E1 !important;
    font-size: 14px;
    margin-top: 4px;
}}
.portal-badge {{
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.22);
    color: #FFFFFF !important;
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 12.5px;
    font-weight: 600;
}}

.step-header {{
    background: #EFF6FF;
    border-left: 4px solid #1D4ED8;
    color: #1D4ED8 !important;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin: 10px 0 10px;
}}

.tier-card {{
    border-radius: 14px;
    padding: 20px 22px;
    margin: 4px 0 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}}
.tier-badge {{ font-size: 11.5px; font-weight: 800; letter-spacing: 1.3px; text-transform: uppercase; }}
.tier-score {{ color: {TEXT} !important; font-size: 34px; font-weight: 800; margin: 6px 0; line-height: 1.1; }}
.tier-score span {{ color: {MUTED} !important; font-size: 14px; font-weight: 500; }}
.tier-desc {{ color: {TEXT} !important; font-size: 14px; line-height: 1.45; }}

.flag {{
    background: {CARD};
    border: 1px solid {BORDER_SOFT};
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    color: {TEXT};
    font-size: 13.5px;
}}
.flag .pts {{
    float: right;
    font-weight: 700;
    font-size: 12.5px;
    padding: 2px 8px;
    border-radius: 6px;
}}
.flag .cav {{ display: block; color: {MUTED}; font-size: 11.5px; margin-top: 4px; }}

.sidebar-pill {{
    background: {SUBCARD};
    border: 1px solid {BORDER_SOFT};
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
}}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL + ASSETS
# --------------------------------------------------
@st.cache_resource(show_spinner="Initializing ClaimWise AI engine ...")
def get_engine():
    model, assets, name = load_assets()
    return model, assets, name, load_background(assets), load_triage(assets, name)


try:
    model, assets, model_name, background, triage = get_engine()
except FileNotFoundError as err:
    st.error(f"Model files not found ({err}). Run `python run_all.py` first, then restart the app.")
    st.stop()

opts, defaults = assets["options"], assets["defaults"]
TYPICAL_SCORE = float(risk_scores(model, background).mean())


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def esc(s):
    return str(s).replace("$", "\\$")


def score_of(X):
    return float(risk_scores(model, X)[0])


def tier_label(key):
    return f"{TIERS[key]['icon']} {TIERS[key]['label']}"


MISSING_TEXT = {"authorities_contacted": "None / Not Notified", "police_report_available": "Unknown",
                "property_damage": "Unknown", "collision_type": "Not Applicable / Unknown"}


def fmt_option(col):
    return lambda v: MISSING_TEXT.get(col, "Not recorded") if v == "Missing" else v


def style_chart(chart):
    return (
        chart.configure(background=CARD)
        .configure_axis(labelColor=TEXT, titleColor=TEXT, gridColor=BORDER_SOFT, domainColor=BORDER)
        .configure_legend(labelColor=TEXT, titleColor=TEXT)
        .configure_view(strokeWidth=0)
    )


def apply_changes(X, changes=None, scale=1.0):
    Xc = X.copy()
    for c, v in (changes or {}).items():
        Xc[c] = np.nan if is_missing(v) else v
    if scale != 1.0:
        for c in CLAIM_PARTS + ["total_claim_amount"]:
            Xc[c] = X[c].astype(float) * scale
    Xc["claim_per_premium"] = Xc["total_claim_amount"] / Xc["policy_annual_premium"].replace(0, np.nan)
    return coerce_types(Xc, assets)


# --------------------------------------------------
# FNOL FORM SPECIFICATION (REAL-WORLD INSURANCE TERMS)
# --------------------------------------------------
def F(col, label, kind, lo=None, hi=None, step=None, help=None):
    return dict(col=col, label=label, kind=kind, lo=lo, hi=hi, step=step, help=help)


STEP1_SPECS = [
    F("incident_type", "What happened? (Incident Type)", "cat"),
    F("incident_severity", "Extent of vehicle damage", "cat"),
    F("collision_type", "Point of impact / collision", "cat_missing"),
    F("authorities_contacted", "Emergency authorities notified", "cat_missing"),
    F("police_report_available", "Was a police report filed?", "cat_missing"),
    F("property_damage", "Third-party property damage?", "cat_missing"),
    F("witnesses", "Number of witnesses", "int", 0, 20),
    F("bodily_injuries", "Number of people injured", "int", 0, 20),
    F("number_of_vehicles_involved", "Vehicles involved", "int", 1, 20),
    F("incident_hour_of_the_day", "Hour of incident (0–23h)", "int", 0, 23),
]

STEP2_SPECS = [
    F("policy_annual_premium", "Annual policy premium ($)", "float", 1.0, 100000.0, 50.0),
    F("policy_deductable", "Policy deductible ($)", "int", 0, 100000, 500),
    F("months_as_customer", "Customer tenure (months)", "int", 0, 1200),
    F("policy_state", "Policy state", "cat"),
    F("policy_csl", "Coverage limit (CSL)", "cat"),
    F("auto_make", "Vehicle make", "cat"),
    F("auto_model", "Vehicle model", "cat"),
    F("auto_year", "Vehicle year", "int", 1980, 2026),
]

STEP3_SPECS = [
    F("injury_claim", "Bodily injury claim ($)", "int", 0, 10_000_000, 500),
    F("property_claim", "Property damage claim ($)", "int", 0, 10_000_000, 500),
    F("vehicle_claim", "Vehicle repair / loss ($)", "int", 0, 10_000_000, 500),
]

MAIN_SPECS = STEP1_SPECS + STEP2_SPECS + STEP3_SPECS
ADV_STEPS = {"umbrella_limit": 1_000_000, "capital-gains": 100, "capital-loss": 100}
_skip = set(ENGINEERED_COLS) | {"total_claim_amount"} | {s["col"] for s in MAIN_SPECS}
ADVANCED = [F(c, FEATURE_LABELS.get(c, c), "cat" if c in opts else "int", step=ADV_STEPS.get(c, 1))
            for c in assets["columns"] if c not in _skip]
ALL_SPECS = MAIN_SPECS + ADVANCED
DATE_MIN, DATE_MAX = date(1950, 1, 1), date(2035, 12, 31)


def field_options(spec):
    base = opts[spec["col"]]
    return ["Missing"] + base if spec["kind"] == "cat_missing" else list(base)


def to_widget_value(spec, v):
    col, kind = spec["col"], spec["kind"]
    if kind in ("cat", "cat_missing"):
        options = field_options(spec)
        if not is_missing(v) and str(v) in options:
            return str(v)
        if kind == "cat_missing":
            return "Missing"
        d = str(defaults.get(col))
        return d if d in options else options[0]
    x = float(defaults.get(col, 0)) if is_missing(v) else float(v)
    if spec["lo"] is not None:
        x = max(x, spec["lo"])
    if spec["hi"] is not None:
        x = min(x, spec["hi"])
    return int(round(x)) if kind == "int" else round(x, 2)


def dates_from_row(row):
    m = row.get("incident_month")
    d = row.get("incident_dayofweek")
    m = 1 if is_missing(m) else int(m)
    d = 0 if is_missing(d) else int(d)
    incident = date(2015, m, 1)
    while incident.weekday() != d:
        incident += timedelta(days=1)
    age = row.get("policy_age_days")
    age = 0 if is_missing(age) else max(int(age), 0)
    return incident - timedelta(days=age), incident


def load_row_into_form(row):
    for spec in ALL_SPECS:
        ss[f"f_{spec['col']}"] = to_widget_value(spec, row.get(spec["col"]))
    ss["f_bind"], ss["f_incident"] = dates_from_row(row)


def init_form(force=False):
    if force or "f_bind" not in ss:
        load_row_into_form(defaults)
        for c in CLAIM_PARTS:
            ss[f"f_{c}"] = 0
        ss["f_incident"] = date.today()
        ss["f_bind"] = date.today() - timedelta(days=365)
        return
    for spec in ALL_SPECS:
        key = f"f_{spec['col']}"
        if key not in ss:
            ss[key] = to_widget_value(spec, defaults.get(spec["col"]))


def render_field(spec):
    key, kind = f"f_{spec['col']}", spec["kind"]
    if kind in ("cat", "cat_missing"):
        st.selectbox(spec["label"], field_options(spec), key=key,
                     format_func=fmt_option(spec["col"]), help=spec["help"])
    elif kind == "int":
        st.number_input(spec["label"], min_value=spec["lo"], max_value=spec["hi"],
                        step=spec["step"] or 1, key=key, help=spec["help"])
    else:
        st.number_input(spec["label"], min_value=spec["lo"], max_value=spec["hi"],
                        step=spec["step"] or 10.0, format="%.2f", key=key, help=spec["help"])


init_form()


# --------------------------------------------------
# REALISTIC DEMO SCENARIOS (HELD-OUT TEST SET)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def get_presets(_model, cache_key):
    path = PROCESSED_DIR / "test.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    y = df.pop(TARGET).to_numpy()
    X = coerce_types(df[assets["columns"]], assets)
    keep = (X["policy_age_days"].fillna(0) >= 0).to_numpy()
    X, y = X[keep].reset_index(drop=True), y[keep]
    s = risk_scores(_model, X)
    tiers = np.array([tier_of(v, triage) for v in s])
    picks = {}
    fraud, genuine = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(fraud):
        picks["high"] = fraud[np.argmax(s[fraud])]
    if len(genuine):
        picks["low"] = genuine[np.argmin(s[genuine])]
    medium = np.where(tiers == "MEDIUM")[0]
    pool = medium if len(medium) else np.arange(len(s))
    picks["medium"] = pool[np.argmin(np.abs(s[pool] - (triage["low"] + triage["high"]) / 2))]
    return {k: {"row": X.iloc[i].to_dict(), "actual": "Fraud" if y[i] == 1 else "Genuine",
                "score": float(s[i])} for k, i in picks.items()}


presets = get_presets(model, f"{model_name}-{triage['low']}")
PRESET_META = {
    "low": ("🚗 Routine Fender-Bender", "Loads a low-risk genuine claim from the held-out test set"),
    "medium": ("⚖️ Borderline Collision", "Loads a medium-risk claim in the desk-review zone"),
    "high": ("🚨 Suspicious Major Claim", "Loads a high-risk fraudulent claim from the held-out test set"),
}


def load_preset(key):
    p = presets[key]
    load_row_into_form(p["row"])
    ss["preset_note"] = (
        f"Loaded a real held-out test claim (Ground Truth: **{p['actual']}**). "
        "The model never saw this claim during training."
    )
    ss["auto_analyze"] = True


def reset_form():
    init_form(force=True)
    for k in ("case", "preset_note"):
        ss.pop(k, None)


# --------------------------------------------------
# WHAT-IF & ANALYSIS
# --------------------------------------------------
WHATIF_FIELDS = {"w_police": "police_report_available", "w_authorities": "authorities_contacted",
                 "w_damage": "property_damage", "w_severity": "incident_severity"}


def reset_whatif():
    if "case" not in ss:
        return
    r = ss["case"]["X"].iloc[0]
    for key, col in WHATIF_FIELDS.items():
        v = r[col]
        if col == "incident_severity":
            ss[key] = str(v) if not is_missing(v) and str(v) in opts[col] else opts[col][0]
        else:
            ss[key] = "Missing" if is_missing(v) else str(v)
    ss["w_witnesses"] = int(min(max(0 if is_missing(r["witnesses"]) else r["witnesses"], 0), 10))
    ss["w_scale"] = 1.0


def analyze():
    bind, incident = ss["f_bind"], ss["f_incident"]
    if incident < bind:
        st.error("Incident date cannot be earlier than the policy start date.")
        ss.pop("case", None)
        return
    inputs = {s["col"]: ss[f"f_{s['col']}"] for s in ALL_SPECS}
    if sum(float(inputs[c]) for c in CLAIM_PARTS) <= 0:
        st.error("Enter at least one claim amount greater than zero.")
        ss.pop("case", None)
        return
    inputs["policy_bind_date"], inputs["incident_date"] = bind.isoformat(), incident.isoformat()
    X = build_row(assets, inputs)
    score, expl = explain_claim(model, X, background, risk_scores)
    ss["case"] = {"X": X, "score": score, "tier": tier_of(score, triage), "expl": expl}
    reset_whatif()


def counterfactuals(X, score):
    r = X.iloc[0]
    cands = []

    def same(cur, new):
        return (is_missing(cur) and is_missing(new)) or (not is_missing(cur) and str(cur) == str(new))

    for col, values in [("police_report_available", ["YES", "NO", None]),
                        ("property_damage", ["YES", "NO", None]),
                        ("authorities_contacted", opts["authorities_contacted"] + [None]),
                        ("incident_severity", opts["incident_severity"]),
                        ("witnesses", [0, 1, 2, 3])]:
        for v in values:
            if not same(r[col], v):
                cands.append((f"{FEATURE_LABELS[col]} → {fmt_value(col, np.nan if v is None else v)}",
                              {col: v}, 1.0))
    for s in [0.5, 0.75, 1.25, 1.5]:
        cands.append((f"All claim amounts ×{s:g}", {}, s))

    big = pd.concat([apply_changes(X, ch, sc) for _, ch, sc in cands], ignore_index=True)
    new = risk_scores(model, coerce_types(big, assets))
    return pd.DataFrame({
        "If this evidence changes ...": [c[0] for c in cands],
        "New risk (%)": new * 100,
        "Change (pts)": (new - score) * 100,
        "New tier": [tier_label(tier_of(v, triage)) for v in new],
    })


def case_report(X, score, tier, expl):
    t = TIERS[tier]
    r = X.iloc[0]
    up = expl[expl["impact"] > 0.005].head(6)
    down = expl[expl["impact"] < -0.005].head(6)
    lines = ["CLAIMWISE AI - ADJUSTER ASSESSMENT DOSSIER", "=" * 60,
             f"Generated : {datetime.now():%Y-%m-%d %H:%M}",
             f"Engine    : {model_name}",
             f"Risk score: {score:.1%}   (typical claim: {TYPICAL_SCORE:.1%})",
             f"Triage    : {t['label']} - {t['queue']}", "",
             "RECOMMENDED ACTIONS"] + [f"  - {a}" for a in t["actions"]] + [
             "", "FACTORS RAISING RISK"] + (
             [f"  + {x.feature}: {x.value}  ({x.impact * 100:+.1f} pts)" + (f"  [{x.caveat}]" if x.caveat else "")
              for x in up.itertuples()] or ["  (none significant)"]) + [
             "", "FACTORS LOWERING RISK"] + (
             [f"  - {x.feature}: {x.value}  ({x.impact * 100:+.1f} pts)" + (f"  [{x.caveat}]" if x.caveat else "")
              for x in down.itertuples()] or ["  (none significant)"]) + [
             "", "CLAIM SNAPSHOT"] + [f"  {FEATURE_LABELS.get(c, c):<26}: {fmt_value(c, r[c])}" for c in X.columns] + [
             "", "Decision-support only. A licensed claims adjuster makes the final determination."]
    return "\n".join(lines)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ ClaimWise AI")
    st.caption("Enterprise Claims & Fraud Portal")
    st.markdown("---")
    nested = assets.get("best", {}).get("nested_f1")
    nested_line = f"<br><span style='color:{MUTED};'>Nested-CV F1:</span> <b>{nested[0]:.3f} ± {nested[1]:.3f}</b>" if nested else ""
    st.markdown(
        f"<div class='sidebar-pill'><b>Production Engine</b><br>{model_name}{nested_line}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Triage Routing Rules")
    st.markdown(
        f"<div class='sidebar-pill'>"
        f"🟢 <b>Fast-Track:</b> &lt; {triage['low']:.0%}<br>"
        f"🟡 <b>Desk Review:</b> {triage['low']:.0%} – {triage['high']:.0%}<br>"
        f"🔴 <b>SIU Hold:</b> ≥ {triage['high']:.0%}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Human-in-the-loop decision support. No claim is denied automatically without adjuster review.")

# --------------------------------------------------
# TOP PORTAL HEADER
# --------------------------------------------------
st.markdown(f"""
<div class="portal-nav">
    <div>
        <div class="portal-eyebrow">AUTO INSURANCE CLAIMS & SPECIAL INVESTIGATION PORTAL</div>
        <div class="portal-title">🛡️ ClaimWise AI — Claims Triage & Intake</div>
        <div class="portal-sub">Submit a First Notice of Loss (FNOL), inspect automated fraud risk drivers, or triage bulk claim batches.</div>
    </div>
    <div class="portal-badge">🟢 Model Online · {model_name}</div>
</div>
""", unsafe_allow_html=True)

tab_portal, tab_batch, tab_model = st.tabs(["🛡️ FNOL Claim Portal", "📂 Batch Claims Queue", "📊 Model & Validation"])

# ==================================================
# TAB 1 - REAL-WORLD FNOL PORTAL & ADJUSTER DOSSIER
# ==================================================
with tab_portal:
    # Scenario toolbar
    if presets:
        st.caption("Quick Demo Scenarios (real claims from the held-out test set):")
    pc = st.columns(4)
    for col, key in zip(pc[:3], ["low", "medium", "high"]):
        if key in presets:
            label, tip = PRESET_META[key]
            col.button(label, key=f"preset_{key}", on_click=load_preset, args=(key,), help=tip)
    pc[3].button("↺ New Blank Claim", key="reset_form", on_click=reset_form)
    if ss.get("preset_note"):
        st.info(ss["preset_note"])

    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    # ---------- LEFT: GUIDED 3-STEP FNOL INTAKE ----------
    with left_col:
        st.markdown("### 📝 First Notice of Loss (FNOL) Intake")
        with st.form("claim_form"):
            st.markdown('<div class="step-header">Step 1 · Incident & Evidence</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            for i, spec in enumerate(STEP1_SPECS):
                with (c1 if i % 2 == 0 else c2):
                    render_field(spec)

            st.markdown('<div class="step-header">Step 2 · Policy & Vehicle Information</div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.date_input("Policy effective date", key="f_bind", min_value=DATE_MIN, max_value=DATE_MAX)
            d2.date_input("Date of incident", key="f_incident", min_value=DATE_MIN, max_value=DATE_MAX)
            p1, p2 = st.columns(2)
            for i, spec in enumerate(STEP2_SPECS):
                with (p1 if i % 2 == 0 else p2):
                    render_field(spec)

            st.markdown('<div class="step-header">Step 3 · Claimed Loss Breakdown ($)</div>', unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            for col_w, spec in zip([a1, a2, a3], STEP3_SPECS):
                with col_w:
                    render_field(spec)

            with st.expander("Additional Policyholder Profile (Optional / Underwriting Fields)"):
                st.caption("Pre-filled with typical portfolio values. Note: `Insured hobby` is a known public dataset artifact.")
                ec1, ec2 = st.columns(2)
                for i, spec in enumerate(ADVANCED):
                    with (ec1 if i % 2 == 0 else ec2):
                        render_field(spec)

            submitted = st.form_submit_button("🔍 Submit Claim for AI Triage")

    auto = ss.pop("auto_analyze", False)
    if submitted or auto:
        if submitted:
            ss.pop("preset_note", None)
        analyze()

    # ---------- RIGHT: ADJUSTER TRIAGE DOSSIER ----------
    with right_col:
        st.markdown("### 📋 Adjuster Triage Dossier")

        if "case" not in ss:
            with st.container(border=True):
                st.markdown("#### No Claim Analyzed Yet")
                st.write(
                    "Enter the claim details and loss amounts on the left and click **Submit Claim for AI Triage**, "
                    "or select one of the **Quick Demo Scenarios** above (loaded from the held-out test dataset) "
                    "to generate a live risk assessment."
                )
        else:
            case = ss["case"]
            X, score, tier, expl = case["X"], case["score"], case["tier"], case["expl"]
            t, stats, ts = TIERS[tier], triage["stats"][tier], TIER_STYLES[tier]
            row0 = X.iloc[0]
            total_amt = float(row0["total_claim_amount"])
            ded_amt = float(row0["policy_deductable"])
            net_amt = max(total_amt - ded_amt, 0.0)
            ratio = float(row0["claim_per_premium"])

            st.markdown(
                f'<div class="tier-card" style="background:{ts["bg"]}; border:1px solid {ts["border"]}; border-left:8px solid {ts["accent"]};">'
                f'<div class="tier-badge" style="color:{ts["fg"]};">{t["icon"]} {ts["status"]}</div>'
                f'<div class="tier-score">{score:.1%} <span>fraud risk score</span></div>'
                f'<div class="tier-desc">{t["summary"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0.0), 1.0))

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Claimed", f"${total_amt:,.0f}")
            m2.metric("Net Exposure", f"${net_amt:,.0f}", f"-${ded_amt:,.0f} deductible", delta_color="off")
            m3.metric("Claim / Premium", f"{ratio:.1f}×", f"Portfolio avg risk: {TYPICAL_SCORE:.1%}", delta_color="off")

            with st.container(border=True):
                st.markdown("**Adjuster Action Checklist**")
                st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(t["actions"], 1)))

            d_tab1, d_tab2, d_tab3 = st.tabs(["🚩 Why Flagged (XAI)", "🧪 What-If Simulator", "📈 Sensitivity & Report"])

            with d_tab1:
                def flag_html(row, sign):
                    fg = "#991B1B" if sign > 0 else "#065F46"
                    bg_pill = "#FEE2E2" if sign > 0 else "#D1FAE5"
                    cav = f'<span class="cav">⚠️ {row.caveat}</span>' if row.caveat else ""
                    return (
                        f'<div class="flag">'
                        f'<span class="pts" style="color:{fg}; background:{bg_pill};">{row.impact * 100:+.1f} pts</span>'
                        f'<b>{row.feature}</b> — {esc(row.value)}{cav}</div>'
                    )

                st.markdown("**🚩 Red Flags (Raise Risk)**")
                up = expl[expl["impact"] > 0.005].head(4)
                st.markdown("".join(flag_html(r, 1) for r in up.itertuples()) or "No significant red flags.",
                            unsafe_allow_html=True)

                st.markdown("**🟢 Mitigating Factors (Lower Risk)**")
                down = expl[expl["impact"] < -0.005].head(4)
                st.markdown("".join(flag_html(r, -1) for r in down.itertuples()) or "No significant mitigating factors.",
                            unsafe_allow_html=True)

                top = expl.head(8).copy()
                top["pts"] = top["impact"] * 100
                top["direction"] = np.where(top["pts"] >= 0, "Raises risk", "Lowers risk")
                top["label"] = top["feature"] + ": " + top["value"]
                bars = alt.Chart(top).mark_bar(cornerRadius=4).encode(
                    x=alt.X("pts:Q", title="Impact on risk score (pts)"),
                    y=alt.Y("label:N", sort=list(top["label"]), title=None, axis=alt.Axis(labelLimit=240)),
                    color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="bottom"),
                                    scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=["#DC2626", "#059669"])),
                    tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("value:N", title="Value"),
                             alt.Tooltip("pts:Q", title="Impact (pts)", format="+.1f")],
                ).properties(width="container", height=270)
                st.altair_chart(style_chart(bars), theme=None)

            with d_tab2:
                st.caption("Test how new investigation findings would change the risk score and triage decision.")
                w1, w2 = st.columns(2)
                with w1:
                    st.selectbox("Police report", ["Missing"] + opts["police_report_available"],
                                 key="w_police", format_func=fmt_option("police_report_available"))
                    st.selectbox("Authorities", ["Missing"] + opts["authorities_contacted"],
                                 key="w_authorities", format_func=fmt_option("authorities_contacted"))
                    st.slider("Witnesses", min_value=0, max_value=10, step=1, key="w_witnesses")
                with w2:
                    st.selectbox("Property damage", ["Missing"] + opts["property_damage"],
                                 key="w_damage", format_func=fmt_option("property_damage"))
                    st.selectbox("Damage severity", opts["incident_severity"], key="w_severity")
                    st.slider("Scale claim amounts", min_value=0.25, max_value=2.0, step=0.05,
                              key="w_scale", format="%.2fx")

                changes = {col: ss[key] for key, col in WHATIF_FIELDS.items()}
                changes["witnesses"] = ss["w_witnesses"]
                Xw = apply_changes(X, changes, ss["w_scale"])
                sw = score_of(Xw)
                tw = tier_of(sw, triage)
                r1, r2, r3 = st.columns(3)
                r1.metric("Original", f"{score:.1%}", tier_label(tier), delta_color="off")
                r2.metric("Simulated", f"{sw:.1%}", f"{(sw - score) * 100:+.1f} pts", delta_color="inverse")
                r3.metric("New Tier", tier_label(tw))
                st.button("↺ Reset Simulator", key="reset_whatif", on_click=reset_whatif)
                if tw != tier:
                    st.warning(f"Decision shifts from **{tier_label(tier)}** → **{tier_label(tw)}**.")

                st.markdown("**Top Single-Fact Counterfactuals**")
                cf = counterfactuals(X, score)
                cfg = {"New risk (%)": st.column_config.NumberColumn(format="%.1f"),
                       "Change (pts)": st.column_config.NumberColumn(format="%+.1f")}
                st.dataframe(
                    pd.concat([
                        cf[cf["Change (pts)"] < -0.05].sort_values("Change (pts)").head(4),
                        cf[cf["Change (pts)"] > 0.05].sort_values("Change (pts)", ascending=False).head(4),
                    ]),
                    hide_index=True, column_config=cfg, use_container_width=True,
                )

            with d_tab3:
                st.markdown("**Claim-Amount Sensitivity Curve**")
                scales = np.round(np.arange(0.25, 2.501, 0.05), 2)
                sens_X = pd.concat([apply_changes(X, None, s) for s in scales], ignore_index=True)
                sens = pd.DataFrame({"total": sens_X["total_claim_amount"].to_numpy(),
                                     "risk": risk_scores(model, sens_X)})
                cur = pd.DataFrame({"total": [float(X["total_claim_amount"].iloc[0])], "risk": [score]})
                thr = pd.DataFrame({"y": [triage["low"], triage["high"]],
                                    "Threshold": ["Desk-review threshold", "SIU threshold"]})
                line = alt.Chart(sens).mark_line(color=PRIMARY, strokeWidth=3).encode(
                    x=alt.X("total:Q", title="Total claim amount ($)", axis=alt.Axis(format="$,.0f")),
                    y=alt.Y("risk:Q", title="Risk score", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
                    tooltip=[alt.Tooltip("total:Q", format="$,.0f", title="Total claim"),
                             alt.Tooltip("risk:Q", format=".1%", title="Risk")])
                rules = alt.Chart(thr).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
                    y="y:Q", color=alt.Color("Threshold:N", title=None, legend=alt.Legend(orient="bottom"),
                                             scale=alt.Scale(range=["#D97706", "#DC2626"])))
                point = alt.Chart(cur).mark_point(size=180, filled=True, color=TEXT).encode(x="total:Q", y="risk:Q")
                st.altair_chart(style_chart(alt.layer(line, rules, point).properties(width="container", height=260)),
                                theme=None)
                st.download_button(
                    "⬇ Download Adjuster Case Dossier (.txt)",
                    case_report(X, score, tier, expl),
                    file_name=f"claim_assessment_{datetime.now():%Y%m%d_%H%M}.txt",
                    mime="text/plain",
                )

# ==================================================
# TAB 2 - BATCH CLAIMS QUEUE
# ==================================================
def run_batch(df_raw, source):
    X, ids, y, notes = prepare_batch(df_raw, assets)
    s = risk_scores(model, X)
    tiers = [tier_of(v, triage) for v in s]
    res = pd.DataFrame({
        "Claim ID": ids,
        "Risk score (%)": s * 100,
        "Tier": [tier_label(k) for k in tiers],
        "tier_key": tiers,
        "Total claim ($)": X["total_claim_amount"].to_numpy(),
        "Incident type": X["incident_type"].fillna("Not recorded").to_numpy(),
        "Severity": X["incident_severity"].fillna("Not recorded").to_numpy(),
        "Police report": X["police_report_available"].fillna("Not recorded").to_numpy(),
    })
    if y is not None:
        res["Actual"] = np.where(np.asarray(y) == 1, "Fraud", "Genuine")
    drivers = {}
    high_idx = res[res["tier_key"] == "HIGH"].sort_values("Risk score (%)", ascending=False).index[:30]
    small_bg = background.head(25)
    for i in high_idx:
        _, ex = explain_claim(model, X.iloc[[i]], small_bg, risk_scores)
        pos = ex[ex["impact"] > 0].head(2)
        drivers[i] = "; ".join(f"{r.feature}: {r.value}" for r in pos.itertuples())
    res["Top risk drivers"] = [drivers.get(i, "") for i in res.index]
    ss["batch"] = {"res": res.sort_values("Risk score (%)", ascending=False).reset_index(drop=True),
                   "notes": notes, "source": source}


with tab_batch:
    st.markdown("## Batch Claims Triage Queue")
    st.caption(esc("Upload a CSV of daily claims to score, route into operational queues, and rank highest-risk claims first."))
    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        up_file = st.file_uploader("Upload Claims CSV", type=["csv"], key="batch_file")
    test_path = PROCESSED_DIR / "test.csv"
    with b2:
        st.write("")
        st.write("")
        if st.button("Score Uploaded File", key="score_upload", disabled=up_file is None):
            with st.spinner("Scoring claims ..."):
                run_batch(pd.read_csv(up_file), up_file.name)
    with b3:
        st.write("")
        st.write("")
        if test_path.exists() and st.button("Load Sample (200 Test Claims)", key="score_sample"):
            with st.spinner("Scoring claims ..."):
                run_batch(pd.read_csv(test_path), "held-out test set (200 claims, with ground-truth labels)")
    if test_path.exists():
        st.download_button("⬇ Download Sample CSV Template", pd.read_csv(test_path).drop(
            columns=[TARGET]).head(25).to_csv(index=False), file_name="sample_claims.csv", mime="text/csv",
            key="dl_sample")

    if "batch" in ss:
        res, notes = ss["batch"]["res"], ss["batch"]["notes"]
        st.markdown("---")
        st.markdown(f"### Queue Summary — {ss['batch']['source']}")
        for n in notes:
            st.warning(n)
        counts = res["tier_key"].value_counts()
        high, med, low = (int(counts.get(k, 0)) for k in TIER_ORDER)
        siu_value = res.loc[res["tier_key"] == "HIGH", "Total claim ($)"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Claims Scored", f"{len(res)}")
        k2.metric("🔴 SIU Referrals", f"{high}", f"${siu_value:,.0f} on hold", delta_color="off")
        k3.metric("🟡 Desk Reviews", f"{med}")
        k4.metric("🟢 Fast-Tracked", f"{low}", f"{low / max(len(res), 1):.0%} auto-eligible", delta_color="off")

        if "Actual" in res.columns:
            fraud = res["Actual"] == "Fraud"
            n_fraud = int(fraud.sum())
            if n_fraud:
                reviewed = fraud & res["tier_key"].isin(["HIGH", "MEDIUM"])
                siu = res["tier_key"] == "HIGH"
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Frauds Caught in Review", f"{reviewed.sum() / n_fraud:.0%}",
                          f"{int(reviewed.sum())} of {n_fraud}", delta_color="off")
                e2.metric("SIU Queue Precision", f"{(fraud & siu).sum() / max(int(siu.sum()), 1):.0%}")
                e3.metric("Frauds Missed (Fast-Track)", f"{int((fraud & (res['tier_key'] == 'LOW')).sum())}")
                e4.metric("Fraud Value Held in SIU", f"${res.loc[fraud & siu, 'Total claim ($)'].sum():,.0f}")

        dist = pd.DataFrame({"Tier": [tier_label(k) for k in TIER_ORDER], "Claims": [high, med, low]})
        dist_chart = alt.Chart(dist).mark_bar(cornerRadius=4).encode(
            x=alt.X("Claims:Q"), y=alt.Y("Tier:N", sort=list(dist["Tier"]), title=None),
            color=alt.Color("Tier:N", legend=None, scale=alt.Scale(
                domain=list(dist["Tier"]), range=[TIER_STYLES[k]["accent"] for k in TIER_ORDER])),
            tooltip=["Tier", "Claims"]).properties(width="container", height=150)
        st.altair_chart(style_chart(dist_chart), theme=None)

        show = st.multiselect("Filter by Triage Tier", [tier_label(k) for k in TIER_ORDER],
                              default=[tier_label(k) for k in TIER_ORDER], key="batch_filter")
        view = res[res["Tier"].isin(show)].drop(columns=["tier_key"])
        st.markdown(f"**Prioritized Investigator Worklist** — {len(view)} claims")
        st.dataframe(view, hide_index=True, use_container_width=True, column_config={
            "Risk score (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "Total claim ($)": st.column_config.NumberColumn(format="$%d"),
        })
        st.download_button("⬇ Download Prioritized Worklist (CSV)", res.drop(columns=["tier_key"]).to_csv(index=False),
                           file_name="claim_worklist.csv", mime="text/csv", key="dl_worklist")

# ==================================================
# TAB 3 - MODEL & VALIDATION
# ==================================================
with tab_model:
    st.markdown("## Model & Methodology")
    st.markdown(esc(
        f"**Active Engine:** {model_name} inside a leak-free `scikit-learn` pipeline "
        "(median/constant imputation, standard scaling, one-hot encoding, class-weighted classifier). "
        "Selected via 5×3 Nested Cross-Validation among Logistic Regression, Random Forest, Gradient Boosting, "
        "and SVM (RBF), benchmarked against a stratified dummy baseline."))

    mpath = RESULTS / "test_metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.markdown("#### Held-Out Test Set Performance (Evaluated Once)")
        cols = st.columns(6)
        for c, (k, lab) in zip(cols, [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1-Score"),
                                      ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Accuracy")]):
            if k in m:
                c.metric(lab, f"{m[k]:.3f}")

    st.markdown("#### Triage Policy Thresholds")
    st.caption(esc(f"Thresholds learned from {triage['method']} ({triage['n_train']} training claims)."))
    bounds = {"HIGH": f"≥ {triage['high']:.0%}", "MEDIUM": f"{triage['low']:.0%} – {triage['high']:.0%}",
              "LOW": f"< {triage['low']:.0%}"}
    st.dataframe(pd.DataFrame([{
        "Tier": tier_label(k), "Queue": TIERS[k]["queue"], "Score range": bounds[k],
        "Share of claims": f"{triage['stats'][k]['share']:.0%}",
        "Fraud rate in tier": f"{triage['stats'][k]['fraud_rate']:.0%}",
        "Share of all fraud": f"{triage['stats'][k]['share_of_all_fraud']:.0%}"} for k in TIER_ORDER]),
        hide_index=True, use_container_width=True)

    figs = [("06_model_comparison.png", "Model comparison (repeated CV)"),
            ("07_ablation.png", "Ablation study"),
            ("08_confusion_matrix.png", "Confusion matrix (test)"),
            ("09_roc_pr_curves.png", "ROC and PR curves (test)"),
            ("10_feature_importance.png", "Global feature importance"),
            ("04_fraud_rate_by_category.png", "Fraud rate by category (training EDA)")]
    existing = [(FIGURES / f, cap) for f, cap in figs if (FIGURES / f).exists()]
    for i in range(0, len(existing), 2):
        cols = st.columns(2)
        for c, (p, cap) in zip(cols, existing[i:i + 2]):
            c.image(str(p), caption=cap)

    for fname, title in [("baseline_cv.csv", "Cross-Validation Results (mean ± std)"),
                         ("nested_cv_tuned.csv", "Nested CV After Hyperparameter Tuning"),
                         ("ablation.csv", "Ablation Study Results")]:
        if (TABLES / fname).exists():
            with st.expander(title):
                st.dataframe(pd.read_csv(TABLES / fname), hide_index=True, use_container_width=True)

    st.markdown("#### Limitations & Responsible AI")
    st.markdown(
        "- **Dataset scale:** Trained on a small public dataset (1,000 claims, ~25% fraud); real portfolios are larger and more imbalanced.\n"
        "- **Dataset artifact:** `incident_severity` and `insured_hobbies` dominate the model. The hobby signal (e.g. chess, cross-fit) is a known synthetic artifact and must not be used in production.\n"
        "- **Score calibration:** Scores come from a class-weighted model, representing risk *rankings* rather than calibrated probabilities.\n"
        "- **Human-in-the-loop:** Decision-support only—no claim should be denied automatically without human investigation.")

st.markdown("---")
st.markdown(
    f'<div style="text-align:center; padding:10px; font-size:12px; color:{MUTED};">'
    'ClaimWise AI · Auto Insurance Claims & Fraud Intelligence Portal</div>',
    unsafe_allow_html=True,
)
