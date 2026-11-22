# Insurance Claim Fraud Detection (AIML Project)

Machine-learning pipeline that flags suspicious auto-insurance claims. Built to follow the
8-stage evaluation pipeline: EDA → split → preprocessing → baselines → tuning (nested CV) →
ablation → one-time test evaluation → explainability + demo.

**Live demo:** _paste your Streamlit link here_

## Dataset
Kaggle *Auto Insurance Claims Data* (`insurance_claims.csv`, 1000 claims, 39 columns, target
`fraud_reported`). Put the file in `data/insurance_claims.csv`. About 25 % of claims are fraud (imbalanced).

## Pipeline (and where each stage lives)
| Stage | Script | Output |
|---|---|---|
| 1 EDA (train only) | `src/s1_eda.py` | `results/figures/01-05`, `results/tables/eda_summary.csv` |
| 2 Stratified split, seed 42 | `src/s2_split.py` | `data/processed/`, `split_summary.csv` |
| 3 Preprocessing in a Pipeline | `src/preprocess.py` | no leakage: imputer/scaler/encoder fit per fold |
| 4 Baseline + 4 models, 5×3 repeated CV | `src/s3_baselines.py` | `baseline_cv.csv` (mean ± std) |
| 5 GridSearchCV + nested CV | `src/s4_tuning.py` | `nested_cv_tuned.csv`, `best_params.json` |
| 6 Ablation | `src/s5_ablation.py` | `ablation.csv`, figure 07 |
| 7 Test evaluation (once) | `src/s6_test_eval.py` | metrics, confusion matrix, ROC/PR, error analysis |
| 8 Explainability + demo | `src/s7_explain.py`, `app/app.py` | feature importance, Streamlit app |

Models: Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF).
Imbalance is handled with class weights (SMOTE is also tested in the ablation if `imbalanced-learn` is installed).
Metrics: precision, recall, F1, ROC-AUC, PR-AUC (accuracy alone is misleading on imbalanced data).

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run_all.py                  # runs everything (about 5 minutes)
streamlit run app/app.py           # demo
```
Run a single stage with e.g. `python -m src.s3_baselines`.

## Results
_Fill in after running (copy from `results/tables/`)._

| Model | F1 (CV) | PR-AUC (CV) |
|---|---|---|
| … | … | … |

Final test metrics: _…_

## Reproducibility
Fixed seed (`SEED = 42` in `src/config.py`), stratified split, test set used only in Stage 7,
all preprocessing inside scikit-learn Pipelines.

## Repository layout
```
src/  app/  data/  models/  results/  report/  run_all.py  requirements.txt
```

## Limitations
Small public dataset; fixed 0.5 decision threshold; no cost-sensitive analysis. See the paper in `report/`.
