"""Risk triage: turn the model's fraud score into an operational queue (3 tiers).

Thresholds are learned from OUT-OF-FOLD predictions on the TRAINING set only, so the
held-out test set is never used to tune them.
  * HIGH   (SIU referral)  : score >= 0.5  -> exactly the claims the model itself flags as fraud
  * MEDIUM (desk review)   : low <= score < 0.5
  * LOW    (fast-track)    : score < low, where `low` is chosen so that at most 5 % of the
                             fraudulent training claims would have been fast-tracked.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import SEED
from .models import build_pipeline

DECISION_THRESHOLD = 0.5          # the model's own decision boundary
MAX_FRAUD_FAST_TRACKED = 0.05     # fraction of frauds we accept in the fast-track queue

TIERS = {
    "LOW": {
        "label": "Fast-Track", "icon": "🟢", "color": "#4C9A6A",
        "queue": "Straight-through processing",
        "summary": "Low model risk. Eligible for fast-track settlement after standard document checks.",
        "actions": ["Confirm the policy was active and the loss is covered",
                    "Approve settlement through the standard workflow",
                    "Include in random quality-audit sampling (e.g. 2-5 % of fast-tracked claims)"],
    },
    "MEDIUM": {
        "label": "Desk Review", "icon": "🟡", "color": "#D9A441",
        "queue": "Secondary adjuster review",
        "summary": "Some risk signals. An adjuster should verify key evidence before paying.",
        "actions": ["Request repair estimates, invoices and photos of the damage",
                    "Obtain the police report and witness statements if missing",
                    "Check the claimant's prior claim history",
                    "Decide: settle, or escalate to the Special Investigation Unit"],
    },
    "HIGH": {
        "label": "SIU Referral", "icon": "🔴", "color": "#C0504D",
        "queue": "Special Investigation Unit",
        "summary": "Strong fraud indicators. Hold payment and refer the claim for investigation.",
        "actions": ["Place the payment on hold pending investigation",
                    "Assign an SIU investigator and take a recorded statement",
                    "Verify the incident with third parties (police, repair shop, hospital)",
                    "Document every finding for the audit trail before any decision"],
    },
}
TIER_ORDER = ["HIGH", "MEDIUM", "LOW"]


def risk_scores(model, X):
    """Fraud risk score in [0, 1] for every row (probability, or sigmoid of an SVM margin)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-model.decision_function(X)))


def tier_of(score, triage):
    if score >= triage["high"]:
        return "HIGH"
    if score >= triage["low"]:
        return "MEDIUM"
    return "LOW"


def compute_triage(X_train, y_train, name, params=None, folds=5):
    """Learn the tier thresholds from out-of-fold scores of the tuned model (training data only)."""
    pipe = build_pipeline(name, X_train, params=params)
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    if hasattr(pipe, "predict_proba"):
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    else:
        margin = cross_val_predict(pipe, X_train, y_train, cv=cv, method="decision_function")
        oof = 1.0 / (1.0 + np.exp(-margin))

    y = np.asarray(y_train).astype(int)
    high = DECISION_THRESHOLD
    low = float(np.quantile(oof[y == 1], MAX_FRAUD_FAST_TRACKED)) if y.sum() else high / 2
    if not 0.0 < low < high:
        low = high / 2
    triage = {"low": round(low, 4), "high": high, "n_train": int(len(y)),
              "method": f"{folds}-fold out-of-fold predictions on the training set", "stats": {}}

    total_fraud = max(int(y.sum()), 1)
    for key in TIER_ORDER:
        mask = np.array([tier_of(s, triage) == key for s in oof])
        n = int(mask.sum())
        triage["stats"][key] = {
            "claims": n,
            "share": round(n / len(y), 4),
            "fraud_rate": round(float(y[mask].mean()) if n else 0.0, 4),
            "share_of_all_fraud": round(float(y[mask].sum()) / total_fraud, 4),
        }
    return triage
