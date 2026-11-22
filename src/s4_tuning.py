"""STAGE 5 - Hyper-parameter tuning with GridSearchCV inside NESTED cross-validation (train only)."""
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .config import INNER_FOLDS, MODELS, OUTER_FOLDS, RESULTS, SEED, TUNE_SCORING, ensure_dirs
from .data import get_split
from .models import PARAM_GRIDS, TUNED_MODELS, build_pipeline
from .utils import SCORING, save_json, save_table, slug, summarize_cv, summary_table


def main():
    ensure_dirs()
    X_train, _, y_train, _ = get_split()
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)

    nested_rows, best_params = {}, {}
    for name in TUNED_MODELS:
        print(f"\nTuning {name} ...")
        grid = GridSearchCV(build_pipeline(name, X_train), PARAM_GRIDS[name],
                            scoring=SCORING[TUNE_SCORING], cv=inner, n_jobs=-1)
        # Nested CV: the outer loop gives an honest score for "tune + train" as a whole
        res = cross_validate(grid, X_train, y_train, cv=outer, scoring=SCORING, n_jobs=1)
        nested_rows[name] = summarize_cv(res)

        # Final tuning on the whole training set -> parameters + fitted model
        grid.fit(X_train, y_train)
        best_params[name] = grid.best_params_
        joblib.dump(grid.best_estimator_, MODELS / f"{slug(name)}.joblib")
        print("  best params:", grid.best_params_)

    save_table(summary_table(nested_rows), "nested_cv_tuned")
    save_json(best_params, RESULTS / "best_params.json")

    best_name = max(nested_rows, key=lambda n: nested_rows[n][TUNE_SCORING][0])
    save_json({"name": best_name, "nested_" + TUNE_SCORING: nested_rows[best_name][TUNE_SCORING],
               "params": best_params[best_name]}, RESULTS / "best_model.json")
    print(f"\nBest model by nested-CV {TUNE_SCORING}: {best_name}")


if __name__ == "__main__":
    main()
