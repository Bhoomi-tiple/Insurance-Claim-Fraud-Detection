"""STAGE 2 - Stratified train/test split with a fixed seed. Test set is locked until Stage 6."""
import pandas as pd

from .data import save_split
from .utils import save_table


def main():
    X_tr, X_te, y_tr, y_te = save_split()
    table = pd.DataFrame({
        "Set": ["Train", "Test"],
        "Rows": [len(y_tr), len(y_te)],
        "Fraud": [int(y_tr.sum()), int(y_te.sum())],
        "Fraud %": [f"{y_tr.mean():.1%}", f"{y_te.mean():.1%}"],
    })
    save_table(table, "split_summary")
    print("\nStratified split done (seed fixed). Files: data/processed/train.csv and test.csv")


if __name__ == "__main__":
    main()
