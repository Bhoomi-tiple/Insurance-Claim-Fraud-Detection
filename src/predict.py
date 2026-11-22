"""Turn a few user inputs into a full model-ready row and score it (used by the demo app)."""
import json

import joblib
import numpy as np
import pandas as pd

from .config import MODELS, RESULTS
from .data import ENGINEERED_COLS, add_engineered
from .models import get_scores
from .utils import slug


def load_assets():
    best = json.loads((RESULTS / "best_model.json").read_text())
    model = joblib.load(MODELS / f"{slug(best['name'])}.joblib")
    assets = json.loads((MODELS / "app_assets.json").read_text())
    return model, assets, best["name"]


def predict_claim(model, assets, inputs):
    """inputs: dict with raw fields, incl. policy_bind_date, incident_date (YYYY-MM-DD),
    injury_claim, property_claim, vehicle_claim. Anything missing falls back to a typical value."""
    row = dict(assets["defaults"])
    row.update({k: (np.nan if v == "Missing" else v) for k, v in inputs.items() if k in assets["columns"]})

    total = sum(float(inputs.get(k, row[k])) for k in ["injury_claim", "property_claim", "vehicle_claim"])
    row["total_claim_amount"] = total
    tmp = pd.DataFrame([{"policy_bind_date": inputs["policy_bind_date"], "incident_date": inputs["incident_date"],
                         "total_claim_amount": total, "policy_annual_premium": row["policy_annual_premium"]}])
    row.update(add_engineered(tmp)[ENGINEERED_COLS].iloc[0].to_dict())

    X = pd.DataFrame([row])[assets["columns"]]
    score = float(get_scores(model, X)[0])
    # SVM gives a decision score, not a probability -> squash it to 0-1 with a sigmoid
    proba = score if hasattr(model, "predict_proba") else float(1 / (1 + np.exp(-score)))
    return proba, int(model.predict(X)[0])
