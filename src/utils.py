"""Small helpers shared by all stages: metrics, tables, saving figures."""
import json
import re

import matplotlib
matplotlib.use("Agg")                       # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from .config import FIGURES, REPORT_TABLES, TABLES, ensure_dirs

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def summarize_cv(res):
    """cross_validate output -> {metric: (mean, std)}"""
    return {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std()) for m in SCORING}


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def summary_table(rows, metrics=None, first_col="Model"):
    """rows: {name: {metric: (mean, std)}} -> DataFrame of 'mean ± std' strings."""
    metrics = metrics or list(SCORING)
    out = [{first_col: n, **{METRIC_LABELS[m]: fmt(*r[m]) for m in metrics}} for n, r in rows.items()]
    return pd.DataFrame(out)


def save_table(df, name):
    ensure_dirs()
    df.to_csv(TABLES / f"{name}.csv", index=False)
    tex = df.to_latex(index=False, escape=True).replace("±", r"$\pm$")
    (REPORT_TABLES / f"{name}.tex").write_text(tex, encoding="utf-8")
    print(f"\n=== {name} ===")
    print(df.to_string(index=False))


def savefig(name):
    ensure_dirs()
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved figure: results/figures/{name}.png")


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")
