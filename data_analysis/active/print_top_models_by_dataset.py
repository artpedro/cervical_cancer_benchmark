from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_analysis.active.dataset_regime_utils import dataset_regime, is_mixed_dataset

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = Path("workspace/analysis/test_eval_results/per_checkpoint_test_metrics.csv")
TOP_K = 5

# Ranking priority (desc)
PRIMARY_RANK_METRIC = "test_acc"
SECONDARY_RANK_METRIC = "test_f1"

# Metrics to report as mean +- std across folds
REPORT_METRICS = ["test_acc", "test_f1", "test_prec", "test_rec", "test_spec"]
DATASET_GROUP_MODE = "all"  # all | solo_only | mixed_only


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +- {std:.4f}"


def main() -> None:
    csv_path = INPUT_CSV.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise RuntimeError(f"Input CSV is empty: {csv_path}")

    required = {"dataset", "model", "origin", "fold", PRIMARY_RANK_METRIC, SECONDARY_RANK_METRIC}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "train_dataset" not in df.columns:
        df["train_dataset"] = df["dataset"].astype(str)
    if "dataset_regime" not in df.columns:
        df["dataset_regime"] = df["train_dataset"].astype(str).map(dataset_regime)
    if DATASET_GROUP_MODE == "solo_only":
        df = df[~df["train_dataset"].astype(str).map(is_mixed_dataset)].copy()
    elif DATASET_GROUP_MODE == "mixed_only":
        df = df[df["train_dataset"].astype(str).map(is_mixed_dataset)].copy()
    if df.empty:
        raise RuntimeError("No rows remain after dataset regime filtering.")

    metric_cols = [m for m in REPORT_METRICS if m in df.columns]
    if not metric_cols:
        raise ValueError("None of REPORT_METRICS columns are present in the CSV.")

    grouped = (
        df.groupby(["dataset", "train_dataset", "dataset_regime", "model", "origin"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]

    # Preserve fold count to show robustness of summary.
    fold_count = (
        df.groupby(["dataset", "train_dataset", "dataset_regime", "model", "origin"], as_index=False)["fold"]
        .nunique()
        .rename(columns={"fold": "n_folds"})
    )
    grouped = grouped.merge(
        fold_count,
        on=["dataset", "train_dataset", "dataset_regime", "model", "origin"],
        how="left",
    )

    pr_mean = f"{PRIMARY_RANK_METRIC}_mean"
    sr_mean = f"{SECONDARY_RANK_METRIC}_mean"
    grouped = grouped.sort_values(
        by=["dataset", pr_mean, sr_mean, "model"],
        ascending=[True, False, False, True],
    )

    datasets = grouped["dataset"].astype(str).drop_duplicates().tolist()
    for dataset in datasets:
        ds = grouped[grouped["dataset"].astype(str) == dataset].head(TOP_K).reset_index(drop=True)
        if ds.empty:
            continue

        print("\n" + "=" * 88)
        print(f"DATASET: {dataset} | TOP {TOP_K} MODELS")
        print("=" * 88)

        for rank, row in enumerate(ds.itertuples(index=False), start=1):
            print(
                f"{rank:>2}. {row.model}  | origin={row.origin}  | "
                f"train={row.train_dataset} ({row.dataset_regime})  | folds={int(row.n_folds)}"
            )
            for metric in metric_cols:
                m = float(getattr(row, f"{metric}_mean"))
                s = float(getattr(row, f"{metric}_std"))
                print(f"    - {metric.replace('test_', ''):>6}: {_fmt(m, s)}")
            print()


if __name__ == "__main__":
    main()

