"""STAGE 1 - Data and EDA.  Only the TRAINING part is explored (test set stays untouched)."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import TARGET, ensure_dirs
from .data import get_split
from .utils import save_table, savefig


def main():
    ensure_dirs()
    X_train, _, y_train, _ = get_split()
    df = X_train.assign(**{TARGET: y_train})
    print("Training shape:", df.shape)

    # 1) summary table: type, missing values, unique values, example
    summary = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(1).values,
        "Unique": df.nunique().values,
        "Example": [df[c].dropna().iloc[0] if df[c].notna().any() else "" for c in df.columns],
    })
    save_table(summary, "eda_summary")

    # 2) class balance
    counts = y_train.value_counts().sort_index()
    print(f"\nClass balance: genuine={counts[0]}, fraud={counts[1]} "
          f"({counts[1] / len(y_train):.1%} fraud)")
    plt.figure(figsize=(4.5, 3.5))
    ax = sns.barplot(x=["Genuine", "Fraud"], y=counts.values, hue=["Genuine", "Fraud"], legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.ylabel("Number of claims"); plt.title("Class balance (training set)")
    savefig("01_class_balance")

    # 3) numeric feature histograms
    num_cols = df.select_dtypes("number").columns.drop(TARGET)
    ax = df[num_cols].hist(figsize=(14, 11), bins=20)
    plt.suptitle("Distribution of numeric features", y=1.0)
    savefig("02_numeric_distributions")

    # 4) correlation heatmap (numeric features + target)
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap")
    savefig("03_correlation_heatmap")

    # 5) fraud rate for important categorical features
    cats = [c for c in ["incident_severity", "insured_hobbies", "incident_type", "collision_type",
                        "authorities_contacted", "property_damage"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, c in zip(axes.ravel(), cats):
        rate = df.groupby(c, dropna=False)[TARGET].mean().sort_values()
        rate.index = rate.index.astype(str)
        rate.plot.barh(ax=ax); ax.set_title(f"Fraud rate by {c}"); ax.set_xlabel("fraud rate")
    savefig("04_fraud_rate_by_category")

    # 6) claim amount vs fraud
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x=TARGET, y="total_claim_amount", hue=TARGET, legend=False)
    plt.xticks([0, 1], ["Genuine", "Fraud"]); plt.title("Total claim amount vs fraud")
    savefig("05_claim_amount_vs_fraud")

    # 7) strongest numeric correlations with fraud (printed for your report)
    corr = df.select_dtypes("number").corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\nTop numeric correlations with fraud:\n", corr.head(8).round(3))


if __name__ == "__main__":
    main()
