"""STAGE 7 - Final evaluation on the held-out TEST set. Run this ONCE, after all decisions are made."""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, average_precision_score, classification_report,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from .config import MODELS, RESULTS, TARGET, TABLES, ensure_dirs
from .data import get_split
from .models import get_scores
from .utils import save_json, save_table, savefig, slug


def main():
    ensure_dirs()
    _, X_test, _, y_test = get_split()
    best = json.loads((RESULTS / "best_model.json").read_text())
    name = best["name"]
    model = joblib.load(MODELS / f"{slug(name)}.joblib")        # trained on the TRAIN set only

    scores = get_scores(model, X_test)
    pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
        "mcc": matthews_corrcoef(y_test, pred),
    }
    save_json(metrics, RESULTS / "test_metrics.json")
    save_table(pd.DataFrame([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}]),
               "test_metrics")
    print("\n", classification_report(y_test, pred, target_names=["Genuine", "Fraud"], zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Genuine", "Fraud"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {name}")
    savefig("08_confusion_matrix")

    # ROC and Precision-Recall curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_test, scores, ax=axes[0], name=name)
    axes[0].plot([0, 1], [0, 1], "k--", label="Chance"); axes[0].set_title("ROC curve"); axes[0].legend()
    PrecisionRecallDisplay.from_predictions(y_test, scores, ax=axes[1], name=name)
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", label="Chance (fraud rate)")
    axes[1].set_title("Precision-Recall curve"); axes[1].legend()
    savefig("09_roc_pr_curves")

    # Error analysis: which claims did we get wrong?
    out = X_test.copy()
    out["actual"], out["predicted"], out["fraud_score"] = y_test.values, pred, scores
    wrong = out[out["actual"] != out["predicted"]]
    wrong.to_csv(TABLES / "misclassified_test_claims.csv", index=False)
    fn = out[(out["actual"] == 1) & (out["predicted"] == 0)]
    fp = out[(out["actual"] == 0) & (out["predicted"] == 1)]
    print(f"Missed fraud (false negatives): {len(fn)} | False alarms (false positives): {len(fp)}")
    for col in ["incident_severity", "insured_hobbies"]:
        if col in out.columns:
            print(f"\nMissed frauds by {col}:\n{fn[col].value_counts().head(5).to_string()}")
            print(f"\nFalse alarms by {col}:\n{fp[col].value_counts().head(5).to_string()}")


if __name__ == "__main__":
    main()
