"""Streamlit demo:  streamlit run app/app.py"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `src` can be imported

import streamlit as st

from src.predict import load_assets, predict_claim

st.set_page_config(page_title="Insurance Claim Fraud Detector", page_icon="🔍")
st.title("🔍 Insurance Claim Fraud Detector")
st.caption("Student ML project - a risk score to help investigators prioritise claims, "
           "not a final decision.")


@st.cache_resource
def get_assets():
    return load_assets()


try:
    model, assets, model_name = get_assets()
except FileNotFoundError:
    st.error("Model files not found. Run `python run_all.py` first, then restart the app.")
    st.stop()

opts = assets["options"]
st.write(f"Model in use: **{model_name}**")

c1, c2 = st.columns(2)
with c1:
    severity = st.selectbox("Incident severity", opts["incident_severity"])
    incident_type = st.selectbox("Incident type", opts["incident_type"])
    collision = st.selectbox("Collision type", ["Missing"] + opts["collision_type"])
    authorities = st.selectbox("Authorities contacted", opts["authorities_contacted"])
    police = st.selectbox("Police report available", ["Missing"] + opts["police_report_available"])
    damage = st.selectbox("Property damage", ["Missing"] + opts["property_damage"])
    hobby = st.selectbox("Insured hobby", opts["insured_hobbies"])
with c2:
    age = st.number_input("Age of insured", 18, 90, 38)
    months = st.number_input("Months as customer", 0, 600, 200)
    premium = st.number_input("Annual premium", 100.0, 5000.0, 1250.0)
    injury = st.number_input("Injury claim amount", 0, 200000, 5000)
    prop = st.number_input("Property claim amount", 0, 200000, 8000)
    vehicle = st.number_input("Vehicle claim amount", 0, 400000, 30000)
    witnesses = st.number_input("Witnesses", 0, 10, 1)
    bind = st.date_input("Policy start date", date(2010, 1, 1), min_value=date(1980, 1, 1))
    incident = st.date_input("Incident date", date(2015, 2, 1), min_value=date(1980, 1, 1))

if st.button("Check claim", type="primary"):
    if incident < bind:
        st.warning("Incident date is before the policy start date - please check the dates.")
    inputs = {
        "incident_severity": severity, "incident_type": incident_type, "collision_type": collision,
        "authorities_contacted": authorities, "police_report_available": police, "property_damage": damage,
        "insured_hobbies": hobby, "age": age, "months_as_customer": months, "policy_annual_premium": premium,
        "injury_claim": injury, "property_claim": prop, "vehicle_claim": vehicle, "witnesses": witnesses,
        "policy_bind_date": bind.isoformat(), "incident_date": incident.isoformat(),
    }
    proba, label = predict_claim(model, assets, inputs)
    st.progress(min(max(proba, 0.0), 1.0))
    st.metric("Fraud risk score", f"{proba:.0%}")
    if label == 1:
        st.error("⚠️ Flagged as POSSIBLE FRAUD - recommend manual investigation.")
    else:
        st.success("✅ Looks genuine - no fraud flag.")
