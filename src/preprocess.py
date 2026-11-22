"""Stage 3: preprocessing that lives INSIDE the model pipeline (so no data leakage)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_lists(X, drop_cols=()):
    X = X.drop(columns=list(drop_cols), errors="ignore")
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return num, cat


def build_preprocessor(X_ref, drop_cols=(), scale=True):
    """Numbers: median-impute (+ scale).  Categories: fill 'Missing' + one-hot.
    Everything is fitted only on the training fold when used inside a Pipeline."""
    num, cat = feature_lists(X_ref, drop_cols)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_steps = [("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num), ("cat", Pipeline(cat_steps), cat)],
        remainder="drop")           # any column not listed (e.g. dropped in ablation) is ignored
