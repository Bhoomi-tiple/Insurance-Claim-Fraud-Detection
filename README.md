# Insurance Claim Fraud Detection (AIML Project)

Machine-learning pipeline and **ClaimWise AI** adjuster portal that flags suspicious auto-insurance claims. Built to follow the
8-stage evaluation pipeline: EDA → split → preprocessing → baselines → tuning (nested CV) →
ablation → one-time test evaluation → explainability + deployment-ready demo.

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
| 8 Explainability + demo | `src/s7_explain.py`, `src/s8_export_app_assets.py`, `app/app.py` | feature importance, triage thresholds, Streamlit portal |

Models: Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF).
Imbalance is handled with class weights (SMOTE is also tested in the ablation if `imbalanced-learn` is installed).
Metrics: precision, recall, F1, ROC-AUC, PR-AUC (accuracy alone is misleading on imbalanced data).

## How to run locally
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run_all.py                  # runs all 8 stages (about 2-3 minutes)
streamlit run app/app.py           # launch ClaimWise AI portal
```
Run a single stage with e.g. `python -m src.s3_baselines`.

## Deployment
- **Streamlit Community Cloud:** Push this repository to GitHub (including `models/` and `results/best_model.json`), connect the repo on [share.streamlit.io](https://share.streamlit.io), and set the main file path to `app/app.py`. Python `3.10.11` (`.python-version`) and pinned dependencies (`requirements.txt`) are included.
- **Docker (Render / Railway / Cloud Run):**
  ```bash
  docker build -t claimwise-ai .
  docker run -p 8501:8501 claimwise-ai
  ```

## Results

### Repeated Stratified CV (5×3) & Nested CV Summary
| Model | Baseline F1 (CV) | Baseline PR-AUC (CV) | Nested-CV F1 (Tuned) |
|---|---|---|---|
| Dummy (baseline) | 0.246 ± 0.088 | 0.254 ± 0.031 | — |
| Logistic Regression | 0.678 ± 0.040 | 0.675 ± 0.055 | 0.683 ± 0.038 |
| Random Forest | 0.646 ± 0.046 | 0.589 ± 0.042 | 0.667 ± 0.040 |
| **Gradient Boosting** | **0.672 ± 0.054** | **0.631 ± 0.066** | **0.756 ± 0.032** |
| SVM (RBF) | 0.623 ± 0.037 | 0.564 ± 0.046 | 0.648 ± 0.057 |

### Final Held-Out Test Set Metrics (`Gradient Boosting`, evaluated once)
- **Recall:** `0.857` | **Precision:** `0.636` | **F1-Score:** `0.730`
- **ROC-AUC:** `0.828` | **PR-AUC:** `0.609` | **Accuracy:** `0.845` | **MCC:** `0.639`

## Reproducibility
Fixed seed (`SEED = 42` in `src/config.py`), stratified split, test set used only in Stage 7,
all preprocessing inside scikit-learn Pipelines.

## Repository layout
```
src/  app/  data/  models/  results/  report/  .streamlit/  Dockerfile  run_all.py  requirements.txt
```

## Limitations
Small public dataset; fixed 0.5 decision threshold; no cost-sensitive analysis; `insured_hobbies` acts as a synthetic dataset artifact. See the paper in `report/`.
