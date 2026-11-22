"""STAGE 8a - Explainability: which features matter most? (permutation importance)"""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import MODELS, RESULTS, SEED, ensure_dirs
from .data import get_split
from .utils import save_table, savefig, slug


def main():
    ensure_dirs()
    _, X_test, _, y_test = get_split()
    name = json.loads((RESULTS / "best_model.json").read_text())["name"]
    model = joblib.load(MODELS / f"{slug(name)}.joblib")

    # shuffle one raw column at a time and see how much F1 drops -> works for ANY model
    r = permutation_importance(model, X_test, y_test, scoring="f1", n_repeats=20,
                               random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"Feature": X_test.columns, "F1 drop": r.importances_mean, "Std": r.importances_std})
           .sort_values("F1 drop", ascending=False))
    save_table(imp.head(15).round(4), "feature_importance_top15")

    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["Feature"], top["F1 drop"], xerr=top["Std"], capsize=2)
    plt.xlabel("Drop in F1 when the feature is shuffled"); plt.title(f"Feature importance - {name}")
    savefig("10_feature_importance")


if __name__ == "__main__":
    main()
