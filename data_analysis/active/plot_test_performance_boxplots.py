from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
PER_CHECKPOINT_CSV = Path("workspace/analysis/test_eval_results/per_checkpoint_test_metrics.csv")
OUTPUT_DIR = Path("workspace/analysis/test_eval_results/plots")

# Metrics to plot as model-comparison boxplots (per dataset)
PLOT_METRICS = ["test_acc", "test_f1", "test_prec", "test_rec"]


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def _model_family(model_name: str) -> str:
    name = model_name.lower()
    transformer_tokens = (
        "vit",
        "former",
        "iformer",
        "levit",
        "fastvit",
        "mobilevit",
        "eat",
    )
    if any(token in name for token in transformer_tokens):
        return "transformer"
    return "cnn"


def _ordered_models_by_family_and_metric(df: pd.DataFrame, metric: str) -> tuple[list[str], int]:
    med = (
        df.groupby("model", as_index=False)[metric]
        .median()
        .rename(columns={metric: "median_metric"})
    )
    med["family"] = med["model"].astype(str).map(_model_family)

    # Crescent order (ascending) inside each family.
    cnn = (
        med[med["family"] == "cnn"]
        .sort_values("median_metric", ascending=True)["model"]
        .astype(str)
        .tolist()
    )
    transformer = (
        med[med["family"] == "transformer"]
        .sort_values("median_metric", ascending=True)["model"]
        .astype(str)
        .tolist()
    )
    return cnn + transformer, len(cnn)


def _plot_dataset(df_dataset: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    metrics = [m for m in PLOT_METRICS if m in df_dataset.columns]
    if not metrics:
        raise ValueError(f"No requested metrics found for dataset={dataset_name}")

    models, n_cnn = _ordered_models_by_family_and_metric(
        df_dataset, metric="test_acc" if "test_acc" in metrics else metrics[0]
    )
    n_models = len(models)
    if n_models == 0:
        return

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(max(12, 1.1 * n_models), 4.5 * n_rows),
        constrained_layout=True,
    )
    # Flatten axes regardless of whether it's a scalar axis or ndarray.
    if hasattr(axes, "flat"):
        axes = list(axes.flat)
    else:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = [
            df_dataset.loc[df_dataset["model"] == model, metric].dropna().to_numpy()
            for model in models
        ]
        ax.boxplot(data, tick_labels=models, showfliers=False)
        if 0 < n_cnn < n_models:
            # Visual separator between CNN and Transformer groups.
            ax.axvline(n_cnn + 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(0.02, 0.96, "CNN (asc)", transform=ax.transAxes, va="top", fontsize=9)
            ax.text(0.70, 0.96, "Transformer (asc)", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_title(f"{dataset_name} | {metric}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Test performance distribution by model — {dataset_name}",
        fontsize=13,
    )
    out_path = out_dir / f"{dataset_name}_model_performance_boxplots.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    csv_path = PER_CHECKPOINT_CSV.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Per-checkpoint CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    _require_columns(df, ["dataset", "model"])
    if len(df) == 0:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    out_dir = OUTPUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in sorted(df["dataset"].dropna().astype(str).unique().tolist()):
        ds_df = df.loc[df["dataset"].astype(str) == dataset_name].copy()
        if ds_df.empty:
            continue
        _plot_dataset(ds_df, dataset_name, out_dir)

    print(f"[DONE] Boxplots generated in: {out_dir}")


if __name__ == "__main__":
    main()

