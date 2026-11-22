"""Per-claim ("local") explanations that are faithful to the trained model.

Method - occlusion with a background sample (marginal feature replacement):
for every feature group G we replace the claim's values of G with the values of N typical
training claims, re-score each copy and average.
    impact(G) = score(claim) - mean(score of claim with G replaced)
Positive impact  -> this information pushes the fraud risk UP compared with a typical claim.
Negative impact  -> it pulls the risk DOWN.
The trained pipeline is used as a black box, so this works for any model and reflects exactly
what the model does (it is not a hand-written rule list). Related features are grouped so that
derived columns (e.g. total claim = injury + property + vehicle) are explained together.
"""
import pandas as pd

FEATURE_LABELS = {
    "months_as_customer": "Months as customer", "age": "Age of insured",
    "policy_state": "Policy state", "policy_csl": "Coverage limits (CSL)",
    "policy_deductable": "Deductible", "policy_annual_premium": "Annual premium",
    "umbrella_limit": "Umbrella limit", "insured_sex": "Sex",
    "insured_education_level": "Education", "insured_occupation": "Occupation",
    "insured_hobbies": "Hobby", "insured_relationship": "Relationship",
    "capital-gains": "Capital gains", "capital-loss": "Capital loss",
    "incident_type": "Incident type", "collision_type": "Collision type",
    "incident_severity": "Incident severity", "authorities_contacted": "Authorities contacted",
    "incident_state": "Incident state", "incident_city": "Incident city",
    "incident_hour_of_the_day": "Hour of incident", "number_of_vehicles_involved": "Vehicles involved",
    "property_damage": "Property damage", "bodily_injuries": "Bodily injuries",
    "witnesses": "Witnesses", "police_report_available": "Police report",
    "total_claim_amount": "Total claim", "injury_claim": "Injury claim",
    "property_claim": "Property claim", "vehicle_claim": "Vehicle claim",
    "auto_make": "Car make", "auto_model": "Car model", "auto_year": "Car year",
    "policy_age_days": "Policy age (days)", "incident_month": "Incident month",
    "incident_dayofweek": "Incident weekday", "claim_per_premium": "Claim / premium",
}

GROUPS = {
    "claim_amounts": ("Claim amounts", ["total_claim_amount", "injury_claim", "property_claim",
                                        "vehicle_claim", "claim_per_premium"]),
    "policy_timing": ("Policy timing", ["policy_age_days", "incident_month", "incident_dayofweek"]),
    "vehicle": ("Vehicle", ["auto_make", "auto_model", "auto_year"]),
}

MONEY_COLS = {"policy_deductable", "policy_annual_premium", "umbrella_limit", "capital-gains",
              "capital-loss", "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"}
MISSING_TEXT = {"authorities_contacted": "none / not recorded"}

# Honest caveats shown next to explanations
ARTIFACT_COLS = {"insured_hobbies"}
SENSITIVE_COLS = {"age", "insured_sex", "insured_education_level", "insured_occupation",
                  "insured_relationship"}


def _missing(v):
    if v is None:
        return True
    if isinstance(v, str):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def fmt_value(col, v):
    """Human-readable value of one column."""
    if _missing(v):
        return MISSING_TEXT.get(col, "not recorded")
    if isinstance(v, str):
        return v
    v = float(v)
    if col in MONEY_COLS:
        return f"${v:,.0f}"
    if col == "claim_per_premium":
        return f"{v:.1f}x premium"
    if col == "auto_year":
        return str(int(v))
    if v.is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}"


def feature_groups(columns):
    used, groups = set(), []
    for key, (label, cols) in GROUPS.items():
        cols = [c for c in cols if c in columns]
        if cols:
            groups.append((key, label, cols))
            used |= set(cols)
    for c in columns:
        if c not in used:
            groups.append((c, FEATURE_LABELS.get(c, c), [c]))
    return groups


def describe(key, cols, row):
    if key == "claim_amounts":
        text = f"{fmt_value('total_claim_amount', row.get('total_claim_amount'))} total"
        cpp = row.get("claim_per_premium")
        if not _missing(cpp):
            text += f" ({float(cpp):.0f}x annual premium)"
        return text
    if key == "policy_timing":
        age = row.get("policy_age_days")
        return "policy age unknown" if _missing(age) else f"policy {int(age):,} days old at incident"
    if key == "vehicle":
        return (f"{fmt_value('auto_year', row.get('auto_year'))} {fmt_value('auto_make', row.get('auto_make'))} "
                f"{fmt_value('auto_model', row.get('auto_model'))}")
    return fmt_value(cols[0], row.get(cols[0]))


def caveat(key):
    if key in ARTIFACT_COLS:
        return "Known artifact of this public dataset - not a legitimate basis for a decision."
    if key in SENSITIVE_COLS:
        return "Demographic attribute - check for fairness; should not drive the decision."
    return ""


def explain_claim(model, x_row, background, score_fn):
    """Return (score, DataFrame[key, feature, value, impact, caveat]) sorted by |impact|."""
    x_row = x_row.reset_index(drop=True).iloc[[0]]
    groups = feature_groups(list(x_row.columns))
    nb = len(background)
    repeated = x_row.loc[x_row.index.repeat(nb)].reset_index(drop=True)

    blocks = []
    for _, _, cols in groups:
        block = repeated.copy()
        for c in cols:
            block[c] = background[c].to_numpy()
        blocks.append(block)
    replaced = score_fn(model, pd.concat(blocks, ignore_index=True)).reshape(len(groups), nb).mean(axis=1)
    score = float(score_fn(model, x_row)[0])

    row = x_row.iloc[0]
    out = pd.DataFrame({
        "key": [g[0] for g in groups],
        "feature": [g[1] for g in groups],
        "value": [describe(g[0], g[2], row) for g in groups],
        "impact": score - replaced,
        "caveat": [caveat(g[0]) for g in groups],
    })
    out = out.reindex(out["impact"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return score, out
