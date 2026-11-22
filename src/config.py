"""Central settings. Change numbers here, not inside the stage scripts."""
from pathlib import Path

SEED = 42                      # fixed seed -> same results every run

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "insurance_claims.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
REPORT_TABLES = ROOT / "report" / "tables"

TARGET = "fraud_reported"      # Y / N in the csv
TEST_SIZE = 0.20

# Cross-validation settings
CV_FOLDS, CV_REPEATS = 5, 3    # Stage 4 (repeated stratified CV)
OUTER_FOLDS, INNER_FOLDS = 5, 3  # Stage 5 (nested CV)
TUNE_SCORING = "f1"            # metric used to pick best hyper-parameters

# Columns that only identify a claim (no real signal, would cause over-fitting)
ID_COLS = ["policy_number", "insured_zip", "incident_location"]


def ensure_dirs():
    for p in (PROCESSED_DIR, MODELS, FIGURES, TABLES, REPORT_TABLES):
        p.mkdir(parents=True, exist_ok=True)
