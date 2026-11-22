"""Loading, cleaning and the (single, fixed) train/test split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_RAW, ID_COLS, PROCESSED_DIR, SEED, TARGET, TEST_SIZE, ensure_dirs

ENGINEERED_COLS = ["policy_age_days", "incident_month", "incident_dayofweek", "claim_per_premium"]
CLAIM_COLS = ["total_claim_amount", "injury_claim", "property_claim", "vehicle_claim", "claim_per_premium"]
DEMOGRAPHIC_COLS = ["age", "insured_sex", "insured_education_level",
                    "insured_occupation", "insured_relationship"]


def load_raw(path=DATA_RAW):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'insurance_claims.csv' from Kaggle and put it in the data/ folder.\n"
            "(For a quick code test only: python -m src.make_synthetic_data)")
    return pd.read_csv(path)


def clean(df):
    """Row-by-row cleaning only (nothing is 'learned' from data -> no leakage)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.replace("?", np.nan)                      # '?' means unknown in this dataset
    df = df.drop(columns=[c for c in df.columns if c.startswith("_c")], errors="ignore")
    df = df.drop(columns=ID_COLS, errors="ignore")
    return df


def parse_dates(s):
    """The Kaggle file stores dates as dd-mm-yyyy; the app sends ISO yyyy-mm-dd.
    Parse with explicit formats so ambiguous dates (e.g. 05-01-2015) are never guessed wrongly."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    rest = out.isna()
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], format="%Y-%m-%d", errors="coerce")
    return out


def add_engineered(df):
    """Create new features from dates and amounts, then drop the raw dates."""
    df = df.copy()
    bind = parse_dates(df["policy_bind_date"])
    inc = parse_dates(df["incident_date"])
    df["policy_age_days"] = (inc - bind).dt.days
    df["incident_month"] = inc.dt.month
    df["incident_dayofweek"] = inc.dt.dayofweek
    df["claim_per_premium"] = df["total_claim_amount"] / df["policy_annual_premium"].replace(0, np.nan)
    return df.drop(columns=["policy_bind_date", "incident_date"])


def prepare():
    """Full raw file -> (X, y). y: 1 = fraud, 0 = genuine."""
    df = add_engineered(clean(load_raw()))
    y = (df[TARGET].astype(str).str.upper() == "Y").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def get_split():
    """Stratified 80/20 split with a fixed seed. Identical every time it is called."""
    X, y = prepare()
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def save_split():
    """Write the split to data/processed so it can be shown as evidence."""
    ensure_dirs()
    X_tr, X_te, y_tr, y_te = get_split()
    X_tr.assign(**{TARGET: y_tr}).to_csv(PROCESSED_DIR / "train.csv", index=False)
    X_te.assign(**{TARGET: y_te}).to_csv(PROCESSED_DIR / "test.csv", index=False)
    return X_tr, X_te, y_tr, y_te
