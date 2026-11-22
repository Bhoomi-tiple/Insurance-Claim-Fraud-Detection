"""STAGE 6 - Ablation: remove ONE component at a time from the best model and measure the change."""
import json

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from .config import CV_FOLDS, CV_REPEATS, RESULTS, SEED, ensure_dirs
from .data import CLAIM_COLS, DEMOGRAPHIC_COLS, ENGINEERED_COLS, get_split
from .models import build_pipeline
from .utils import SCORING, save_table, savefig, summarize_cv
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ensure_dirs()
    X_train, _, y_train, _ = get_split()
    best = json.loads((RESULTS / "best_model.json").read_text())
    name, params = best["name"], best["params"]
    print("Ablation on:", name)

    variants = {
        "Full model": dict(),
        "- class weighting (no balancing)": dict(balance="none"),
        "- feature scaling": dict(scale=False),
        "- engineered features": dict(drop_cols=ENGINEERED_COLS),
        "- claim amounts": dict(drop_cols=CLAIM_COLS),
        "- insured hobbies": dict(drop_cols=["insured_hobbies"]),
        "- incident severity": dict(drop_cols=["incident_severity"]),
        "- demographics": dict(drop_cols=DEMOGRAPHIC_COLS),
    }
    try:
        import imblearn  # noqa: F401
        variants["class weighting -> SMOTE"] = dict(balance="smote")
    except ImportError:
        print("imbalanced-learn not installed -> SMOTE variant skipped (pip install imbalanced-learn)")

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED)
    rows, summaries = [], {}
    for label, kwargs in variants.items():
        print("  running:", label)
        pipe = build_pipeline(name, X_train, params=params, **kwargs)
        summaries[label] = summarize_cv(cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1))

    full_f1 = summaries["Full model"]["f1"][0]
    for label, s in summaries.items():
        rows.append({"Variant": label,
                     "F1": f"{s['f1'][0]:.3f} ± {s['f1'][1]:.3f}",
                     "Recall": f"{s['recall'][0]:.3f} ± {s['recall'][1]:.3f}",
                     "PR-AUC": f"{s['pr_auc'][0]:.3f} ± {s['pr_auc'][1]:.3f}",
                     "Δ F1 vs full": f"{s['f1'][0] - full_f1:+.3f}"})
    save_table(pd.DataFrame(rows), "ablation")

    plt.figure(figsize=(8, 4.5))
    labels = list(summaries)
    plt.barh(labels, [summaries[l]["f1"][0] for l in labels],
             xerr=[summaries[l]["f1"][1] for l in labels], capsize=3)
    plt.axvline(full_f1, color="red", linestyle="--", label="Full model")
    plt.gca().invert_yaxis(); plt.xlabel("F1 (mean ± std)"); plt.title(f"Ablation study - {name}")
    plt.legend()
    savefig("07_ablation")


if __name__ == "__main__":
    main()
