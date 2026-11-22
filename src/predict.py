"""Model-serving helpers used by the Streamlit app:
load the model + assets, build model-ready rows, prepare uploaded batches."""
import json

import joblib
import numpy as np
import pandas as pd

from .config import MODELS, PROCESSED_DIR, RESULTS, SEED, TARGET
from .data import ENGINEERED_COLS, add_engineered, clean
from .triage import compute_triage, risk_scores
from .utils import slug

CLAIM_PARTS = ["injury_claim", "property_claim", "vehicle_claim"]
DATE_COLS = ["policy_bind_date", "incident_date"]


def load_assets():
    best = json.loads((RESULTS / "best_model.json").read_text())
    model = joblib.load(MODELS / f"{slug(best['name'])}.joblib")
    assets = json.loads((MODELS / "app_assets.json").read_text())
    assets["best"] = best
    return model, assets, best["name"]


def is_missing(v):
    """True for None / NaN / empty string / the UI placeholder 'Missing'."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "" or v == "Missing"
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def coerce_types(X, assets):
    """Categorical columns -> str/NaN objects, everything else -> numeric (same as training)."""
    X = X.copy()
    cats = set(assets["options"])
    for c in X.columns:
        if c in cats:
            X[c] = X[c].map(lambda v: np.nan if is_missing(v) else str(v)).astype(object)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def build_row(assets, inputs, base=None):
    """inputs: raw fields (incl. policy_bind_date / incident_date as YYYY-MM-DD and the three
    claim parts). Anything not given falls back to `base`, then to a typical training value."""
    cols = assets["columns"]
    row = dict(assets["defaults"])
    if base:
        row.update({k: v for k, v in base.items() if k in cols})
    row.update({k: v for k, v in inputs.items() if k in cols})

    total = sum(float(row[k]) for k in CLAIM_PARTS)
    row["total_claim_amount"] = total
    if all(k in inputs for k in DATE_COLS):
        tmp = pd.DataFrame([{"policy_bind_date": inputs["policy_bind_date"],
                             "incident_date": inputs["incident_date"],
                             "total_claim_amount": total,
                             "policy_annual_premium": float(row["policy_annual_premium"])}])
        row.update(add_engineered(tmp)[ENGINEERED_COLS].iloc[0].to_dict())
    else:
        premium = float(row["policy_annual_premium"])
        row["claim_per_premium"] = total / premium if premium else np.nan
    return coerce_types(pd.DataFrame([row])[cols], assets)


def predict_claim(model, assets, inputs):
    """Backwards-compatible helper: (risk score, predicted label)."""
    X = build_row(assets, inputs)
    return float(risk_scores(model, X)[0]), int(model.predict(X)[0])


def prepare_batch(df_raw, assets):
    """Uploaded CSV (raw Kaggle format or the processed format) -> (X, ids, y or None, notes)."""
    cols, defaults = assets["columns"], assets["defaults"]
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.reset_index(drop=True)

    if "policy_number" in df.columns:
        ids = df["policy_number"].astype(str)
    else:
        ids = pd.Series([f"CLM-{i + 1:04d}" for i in range(len(df))])

    y = None
    if TARGET in df.columns:
        t = df[TARGET]
        if t.dtype == object:
            y = t.astype(str).str.strip().str.upper().isin(["Y", "YES", "1", "TRUE"]).astype(int)
        else:
            y = (pd.to_numeric(t, errors="coerce").fillna(0) > 0).astype(int)
        df = df.drop(columns=[TARGET])

    df = clean(df)
    notes = []
    for c in CLAIM_PARTS + ["total_claim_amount", "policy_annual_premium"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "total_claim_amount" not in df.columns and all(c in df.columns for c in CLAIM_PARTS):
        df["total_claim_amount"] = df[CLAIM_PARTS].sum(axis=1)

    if all(c in df.columns for c in DATE_COLS):
        for c in ["total_claim_amount", "policy_annual_premium"]:
            if c not in df.columns:
                df[c] = defaults[c]
        df = add_engineered(df)
        n_bad = int(df["policy_age_days"].isna().sum())
        if n_bad:
            notes.append(f"{n_bad} row(s) had unreadable dates; their timing features were imputed.")
    if ("claim_per_premium" not in df.columns and "total_claim_amount" in df.columns
            and "policy_annual_premium" in df.columns):
        df["claim_per_premium"] = df["total_claim_amount"] / df["policy_annual_premium"].replace(0, np.nan)

    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = defaults[c]
    if missing:
        notes.append("Columns not found in the file were filled with typical values: " + ", ".join(missing))
    return coerce_types(df[cols], assets), ids.reset_index(drop=True), y, notes


def _load_train_processed(assets):
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    y = train.pop(TARGET)
    return coerce_types(train[assets["columns"]], assets), y


def load_background(assets, n=60):
    """A small sample of typical TRAINING claims used as the reference for explanations."""
    path = MODELS / "background.csv"
    if path.exists():
        bg = pd.read_csv(path).drop(columns=[TARGET], errors="ignore")
        bg = coerce_types(bg[assets["columns"]], assets)
    else:
        bg, _ = _load_train_processed(assets)
    if len(bg) > n:
        bg = bg.sample(n, random_state=SEED)
    return bg.reset_index(drop=True)


def load_triage(assets, name):
    """Triage thresholds saved by Stage 8b; recomputed from the training split if missing."""
    if "triage" in assets:
        return assets["triage"]
    X, y = _load_train_processed(assets)
    return compute_triage(X, y, name, assets.get("best", {}).get("params"))
