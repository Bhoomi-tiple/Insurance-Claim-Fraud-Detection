"""Stage 4/5: the models, the full pipeline builder and the tuning grids."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import SEED
from .preprocess import build_preprocessor

MODEL_NAMES = ["Dummy (baseline)", "Logistic Regression", "Random Forest",
               "Gradient Boosting", "SVM (RBF)"]
TUNED_MODELS = MODEL_NAMES[1:]

PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest": {"clf__n_estimators": [200, 400],
                      "clf__max_depth": [None, 6, 12],
                      "clf__min_samples_leaf": [1, 3]},
    "Gradient Boosting": {"clf__learning_rate": [0.03, 0.1],
                          "clf__max_depth": [3, 5],
                          "clf__max_iter": [100, 200]},
    "SVM (RBF)": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]},
}


def make_model(name, balanced=True):
    cw = "balanced" if balanced else None
    if name == "Dummy (baseline)":
        return DummyClassifier(strategy="stratified", random_state=SEED)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, class_weight=cw, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=SEED, n_jobs=1)
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(class_weight=cw, random_state=SEED)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", class_weight=cw, random_state=SEED)
    raise ValueError(name)


def build_pipeline(name, X_ref, balance="class_weight", scale=True, drop_cols=(), params=None):
    """balance: 'class_weight' | 'smote' | 'none'."""
    pre = build_preprocessor(X_ref, drop_cols, scale)
    clf = make_model(name, balanced=(balance == "class_weight"))
    if balance == "smote":
        from imblearn.over_sampling import SMOTE          # SMOTE only ever sees training folds
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    else:
        pipe = Pipeline([("prep", pre), ("clf", clf)])
    if params:
        pipe.set_params(**params)
    return pipe


def get_scores(pipe, X):
    """Fraud probability (or decision score) for each row."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)
