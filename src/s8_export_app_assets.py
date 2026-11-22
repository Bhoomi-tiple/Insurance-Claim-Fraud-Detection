"""STAGE 8b - Save everything the Streamlit app needs:
default values, dropdown options, triage thresholds and a background sample for explanations.
Uses the TRAINING split only."""
import json

from .config import MODELS, RESULTS, SEED, ensure_dirs
from .data import get_split
from .triage import compute_triage
from .utils import save_json

BACKGROUND_ROWS = 200


def main():
    ensure_dirs()
    X_train, _, y_train, _ = get_split()
    num = X_train.select_dtypes("number")
    cat = X_train.select_dtypes(exclude="number")
    assets = {
        "columns": X_train.columns.tolist(),
        "defaults": {**num.median().to_dict(), **{c: cat[c].mode().iloc[0] for c in cat.columns}},
        "options": {c: sorted(cat[c].dropna().astype(str).unique().tolist()) for c in cat.columns},
        "ranges": {c: [float(num[c].min()), float(num[c].max())] for c in num.columns},
    }

    best = json.loads((RESULTS / "best_model.json").read_text())
    print("Learning triage thresholds from out-of-fold predictions on the training set ...")
    assets["triage"] = compute_triage(X_train, y_train, best["name"], best["params"])
    save_json(assets, MODELS / "app_assets.json")
    print("Saved models/app_assets.json")

    X_train.sample(min(BACKGROUND_ROWS, len(X_train)), random_state=SEED).to_csv(
        MODELS / "background.csv", index=False)
    print("Saved models/background.csv (reference claims for explanations)")

    t = assets["triage"]
    print(f"\nTriage thresholds: fast-track < {t['low']:.3f} <= desk review < {t['high']:.3f} <= SIU")
    for key, s in t["stats"].items():
        print(f"  {key:<6} claims={s['claims']:>3}  share={s['share']:.1%}  fraud rate={s['fraud_rate']:.1%}"
              f"  share of all fraud={s['share_of_all_fraud']:.1%}")


if __name__ == "__main__":
    main()
