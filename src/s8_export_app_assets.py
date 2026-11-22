"""STAGE 8b - Save the default values and dropdown options the Streamlit demo needs."""
from .config import MODELS, ensure_dirs
from .data import get_split
from .utils import save_json


def main():
    ensure_dirs()
    X_train, _, _, _ = get_split()
    num = X_train.select_dtypes("number")
    cat = X_train.select_dtypes(exclude="number")
    assets = {
        "columns": X_train.columns.tolist(),
        "defaults": {**num.median().to_dict(), **{c: cat[c].mode().iloc[0] for c in cat.columns}},
        "options": {c: sorted(cat[c].dropna().astype(str).unique().tolist()) for c in cat.columns},
        "ranges": {c: [float(num[c].min()), float(num[c].max())] for c in num.columns},
    }
    save_json(assets, MODELS / "app_assets.json")
    print("Saved models/app_assets.json")


if __name__ == "__main__":
    main()
