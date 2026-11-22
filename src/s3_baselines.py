"""STAGE 3+4 - Preprocessing pipeline + baseline and 4 models, repeated stratified CV (train only)."""
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from .config import CV_FOLDS, CV_REPEATS, SEED, ensure_dirs
from .data import get_split
from .models import MODEL_NAMES, build_pipeline
from .utils import SCORING, save_json, save_table, savefig, summarize_cv, summary_table
from .config import RESULTS


def main():
    ensure_dirs()
    X_train, _, y_train, _ = get_split()
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED)

    rows = {}
    for name in MODEL_NAMES:
        print(f"Cross-validating {name} ...")
        pipe = build_pipeline(name, X_train)                # preprocessing is inside the pipeline
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows[name] = summarize_cv(res)

    save_table(summary_table(rows), "baseline_cv")
    save_json({n: {m: list(v) for m, v in r.items()} for n, r in rows.items()}, RESULTS / "baseline_cv.json")

    # bar chart of F1 and PR-AUC with error bars
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.38
    xs = range(len(MODEL_NAMES))
    for k, (metric, label) in enumerate([("f1", "F1"), ("pr_auc", "PR-AUC")]):
        ax.bar([x + k * width for x in xs], [rows[n][metric][0] for n in MODEL_NAMES], width,
               yerr=[rows[n][metric][1] for n in MODEL_NAMES], capsize=3, label=label)
    ax.set_xticks([x + width / 2 for x in xs]); ax.set_xticklabels(MODEL_NAMES, rotation=15)
    ax.set_ylabel("Score (mean ± std)"); ax.set_title(f"Model comparison ({CV_FOLDS}-fold x {CV_REPEATS} repeated CV)")
    ax.legend()
    savefig("06_model_comparison")


if __name__ == "__main__":
    main()
