"""Execute Milestone 1 dataset verification and exploratory analysis.

This module does not clean, split, preprocess, or model the data. It records
measured dataset facts and produces a focused set of dissertation-ready EDA
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FIGURES_DIR, METRICS_DIR, RAW_CSV, TARGET_COL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV)
    predictors = [column for column in df.columns if column != TARGET_COL]
    numeric = df.select_dtypes(include=[np.number])

    missing_by_column = df.isna().sum().astype(int)
    infinite_by_column = pd.Series(
        np.isinf(numeric.to_numpy()).sum(axis=0), index=numeric.columns, dtype=int
    )

    duplicate_removed_mask = df.duplicated(keep="first")
    duplicate_all_mask = df.duplicated(keep=False)
    duplicate_removed_by_class = (
        df.loc[duplicate_removed_mask, TARGET_COL]
        .value_counts()
        .sort_index()
        .reindex([0, 1], fill_value=0)
    )

    feature_duplicate_mask = df.duplicated(subset=predictors, keep=False)
    feature_duplicate_rows = df.loc[feature_duplicate_mask]
    if feature_duplicate_rows.empty:
        conflicting_feature_groups = 0
        conflicting_feature_rows = 0
    else:
        target_cardinality = feature_duplicate_rows.groupby(
            predictors, dropna=False, sort=False
        )[TARGET_COL].transform("nunique")
        conflict_mask = target_cardinality > 1
        conflicting_feature_rows = int(conflict_mask.sum())
        conflicting_feature_groups = int(
            feature_duplicate_rows.loc[conflict_mask]
            .drop_duplicates(subset=predictors)
            .shape[0]
        )

    class_counts = (
        df[TARGET_COL].value_counts().sort_index().reindex([0, 1], fill_value=0)
    )
    class_table = pd.DataFrame(
        {
            "class": [0, 1],
            "label": ["Legitimate", "Fraudulent"],
            "count": [int(class_counts.loc[0]), int(class_counts.loc[1])],
            "percentage": [
                float(class_counts.loc[0] / len(df) * 100),
                float(class_counts.loc[1] / len(df) * 100),
            ],
        }
    )

    class_feature_summary = (
        df.groupby(TARGET_COL)[["Amount", "Time"]]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(6)
    )
    class_feature_summary.columns = [
        f"{feature.lower()}_{statistic}"
        for feature, statistic in class_feature_summary.columns
    ]
    class_feature_summary = class_feature_summary.reset_index()

    profile = {
        "source_file": str(RAW_CSV.relative_to(RAW_CSV.parents[2])),
        "sha256": sha256_file(RAW_CSV),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "predictor_columns": int(len(predictors)),
        "column_names": list(df.columns),
        "all_columns_numeric": bool(len(numeric.columns) == len(df.columns)),
        "target_labels": [int(value) for value in sorted(df[TARGET_COL].unique())],
        "legitimate_count": int(class_counts.loc[0]),
        "fraud_count": int(class_counts.loc[1]),
        "fraud_percentage": float(class_counts.loc[1] / len(df) * 100),
        "legitimate_to_fraud_ratio": float(class_counts.loc[0] / class_counts.loc[1]),
        "missing_values_total": int(missing_by_column.sum()),
        "infinite_values_total": int(infinite_by_column.sum()),
        "exact_duplicate_rows_after_first": int(duplicate_removed_mask.sum()),
        "all_rows_in_exact_duplicate_groups": int(duplicate_all_mask.sum()),
        "duplicate_rows_after_first_class_0": int(duplicate_removed_by_class.loc[0]),
        "duplicate_rows_after_first_class_1": int(duplicate_removed_by_class.loc[1]),
        "feature_identical_conflicting_label_groups": conflicting_feature_groups,
        "rows_in_conflicting_label_groups": conflicting_feature_rows,
        "amount_min": float(df["Amount"].min()),
        "amount_median": float(df["Amount"].median()),
        "amount_mean": float(df["Amount"].mean()),
        "amount_max": float(df["Amount"].max()),
        "time_min": float(df["Time"].min()),
        "time_max": float(df["Time"].max()),
        "legitimate_amount_mean": float(df.loc[df[TARGET_COL] == 0, "Amount"].mean()),
        "legitimate_amount_median": float(df.loc[df[TARGET_COL] == 0, "Amount"].median()),
        "fraud_amount_mean": float(df.loc[df[TARGET_COL] == 1, "Amount"].mean()),
        "fraud_amount_median": float(df.loc[df[TARGET_COL] == 1, "Amount"].median()),
    }

    (METRICS_DIR / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    class_table.to_csv(METRICS_DIR / "class_distribution.csv", index=False)
    class_feature_summary.to_csv(METRICS_DIR / "class_feature_summary.csv", index=False)
    df.dtypes.astype(str).rename("dtype").to_csv(
        METRICS_DIR / "column_data_types.csv", header=True
    )
    missing_by_column.rename("missing_count").to_csv(
        METRICS_DIR / "missing_values.csv", header=True
    )
    infinite_by_column.rename("infinite_count").to_csv(
        METRICS_DIR / "infinite_values.csv", header=True
    )
    df.describe().T.to_csv(METRICS_DIR / "descriptive_statistics.csv")
    pd.DataFrame(
        {
            "measure": [
                "exact_duplicate_rows_after_first",
                "all_rows_in_exact_duplicate_groups",
                "duplicate_rows_after_first_class_0",
                "duplicate_rows_after_first_class_1",
                "feature_identical_conflicting_label_groups",
                "rows_in_conflicting_label_groups",
            ],
            "value": [
                profile["exact_duplicate_rows_after_first"],
                profile["all_rows_in_exact_duplicate_groups"],
                profile["duplicate_rows_after_first_class_0"],
                profile["duplicate_rows_after_first_class_1"],
                profile["feature_identical_conflicting_label_groups"],
                profile["rows_in_conflicting_label_groups"],
            ],
        }
    ).to_csv(METRICS_DIR / "duplicate_analysis.csv", index=False)

    correlation_with_target = (
        df.corr(numeric_only=True)[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=np.abs, ascending=False)
        .rename("correlation_with_class")
    )
    correlation_with_target.to_csv(
        METRICS_DIR / "feature_target_correlations.csv", index_label="feature"
    )

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#376A8A", "#C95454"]
    bars = ax.bar(class_table["label"], class_table["count"], color=colors)
    ax.set_title("Transaction Class Distribution")
    ax.set_ylabel("Number of transactions")
    for bar, count in zip(bars, class_table["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
        )
    save_figure(fig, "class_distribution.png")

    amount_cap = df["Amount"].quantile(0.995)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df.loc[df["Amount"] <= amount_cap, "Amount"], bins=70, color="#376A8A")
    ax.set_title("Transaction Amount Distribution up to the 99.5th Percentile")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Number of transactions")
    save_figure(fig, "amount_distribution.png")

    plot_sample = pd.concat(
        [
            df.loc[df[TARGET_COL] == 0].sample(
                n=min(20_000, int((df[TARGET_COL] == 0).sum())), random_state=42
            ),
            df.loc[df[TARGET_COL] == 1],
        ]
    ).copy()
    plot_sample["Class label"] = plot_sample[TARGET_COL].map(
        {0: "Legitimate", 1: "Fraudulent"}
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.boxplot(
        data=plot_sample,
        x="Class label",
        y=np.log1p(plot_sample["Amount"]),
        hue="Class label",
        palette=colors,
        legend=False,
        ax=ax,
    )
    ax.set_title("Transaction Amount by Class")
    ax.set_ylabel("log1p Amount for visualization")
    ax.set_xlabel("")
    save_figure(fig, "amount_by_class.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, color in [(0, colors[0]), (1, colors[1])]:
        values = df.loc[df[TARGET_COL] == label, "Time"]
        ax.hist(
            values,
            bins=60,
            density=True,
            alpha=0.55,
            color=color,
            label="Legitimate" if label == 0 else "Fraudulent",
        )
    ax.set_title("Time Distribution by Class")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, "time_distribution.png")

    selected_features = list(correlation_with_target.head(8).index)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, feature in zip(axes.flat, selected_features):
        for label, color in [(0, colors[0]), (1, colors[1])]:
            values = plot_sample.loc[plot_sample[TARGET_COL] == label, feature]
            ax.hist(
                values,
                bins=45,
                density=True,
                alpha=0.5,
                color=color,
                label="Legitimate" if label == 0 else "Fraudulent",
            )
        ax.set_title(feature)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Distributions of Features Most Correlated with Class", y=1.01)
    fig.tight_layout()
    save_figure(fig, "selected_feature_distributions.png")

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        center=0,
        linewidths=0.05,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    save_figure(fig, "correlation_heatmap.png")

    print(json.dumps(profile, indent=2))
    return profile


if __name__ == "__main__":
    run()
