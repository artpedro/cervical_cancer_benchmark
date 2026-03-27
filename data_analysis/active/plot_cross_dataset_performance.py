from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = Path("workspace/analysis/test_eval_results_cross_dataset")
TEST_SPLIT_CSV = INPUT_DIR / "cross_dataset_testsplit_metrics.csv"
FULL_SCOPE_CSV = INPUT_DIR / "cross_dataset_full_riva_herlev_metrics.csv"

OUTPUT_DIR = INPUT_DIR / "plots"

# Use ascending order inside each family (CNN -> Transformer),
# matching your recent preferred plot style.
ORDER_ASCENDING = True


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


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "source_dataset",
        "source_model",
        "target_dataset",
        "target_acc",
        "target_f1",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["source_dataset"] = out["source_dataset"].astype(str)
    out["source_model"] = out["source_model"].astype(str)
    out["target_dataset"] = out["target_dataset"].astype(str)
    out["source_label"] = out["source_dataset"] + " -> " + out["source_model"]
    out["family"] = out["source_model"].map(_model_family)
    return out


def _ordered_labels(df: pd.DataFrame, metric: str) -> tuple[list[str], int]:
    med = (
        df.groupby("source_label", as_index=False)
        .agg(
            median_metric=(metric, "median"),
            family=("family", "first"),
        )
        .sort_values("median_metric", ascending=ORDER_ASCENDING)
    )
    cnn_labels = med.loc[med["family"] == "cnn", "source_label"].tolist()
    trf_labels = med.loc[med["family"] == "transformer", "source_label"].tolist()
    return cnn_labels + trf_labels, len(cnn_labels)


def _plot_target_boxplots(df: pd.DataFrame, scope_name: str, out_dir: Path) -> None:
    for target in sorted(df["target_dataset"].unique().tolist()):
        dft = df[df["target_dataset"] == target].copy()
        if dft.empty:
            continue

        labels, n_cnn = _ordered_labels(dft, metric="target_acc")
        if not labels:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(max(14, 0.60 * len(labels)), 6), constrained_layout=True)
        metrics = [("target_acc", "Accuracy"), ("target_f1", "F1-score")]

        for ax, (metric, pretty) in zip(axes, metrics):
            data = [dft.loc[dft["source_label"] == lab, metric].dropna().to_numpy() for lab in labels]
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            if 0 < n_cnn < len(labels):
                ax.axvline(n_cnn + 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
                ax.text(0.01, 0.96, "CNN", transform=ax.transAxes, va="top", fontsize=9)
                ax.text(0.73, 0.96, "Transformer", transform=ax.transAxes, va="top", fontsize=9)
            ax.set_title(f"{target} | {pretty}")
            ax.set_ylabel(pretty)
            ax.tick_params(axis="x", rotation=55, labelsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.25)

        fig.suptitle(f"Cross-dataset ({scope_name}) — source model performance on target={target}", fontsize=13)
        out = out_dir / f"{scope_name}__target_{target}__boxplot_acc_f1.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[OK] wrote {out}")


def _plot_transfer_heatmaps(df: pd.DataFrame, scope_name: str, out_dir: Path) -> None:
    agg = (
        df.groupby(["source_label", "target_dataset"], as_index=False)
        .agg(
            mean_acc=("target_acc", "mean"),
            mean_f1=("target_f1", "mean"),
        )
    )
    if agg.empty:
        return

    ordered_labels, _ = _ordered_labels(df, metric="target_acc")
    targets = sorted(agg["target_dataset"].unique().tolist())

    def _pivot(val_col: str) -> pd.DataFrame:
        p = agg.pivot(index="source_label", columns="target_dataset", values=val_col).reindex(index=ordered_labels, columns=targets)
        return p

    for val_col, pretty in [("mean_acc", "Mean Accuracy"), ("mean_f1", "Mean F1-score")]:
        p = _pivot(val_col)
        if p.empty:
            continue
        arr = p.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(targets)), max(8, 0.30 * len(ordered_labels))), constrained_layout=True)
        im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=np.nanmin(arr), vmax=np.nanmax(arr))
        ax.set_xticks(np.arange(len(targets)))
        ax.set_xticklabels(targets, rotation=0)
        ax.set_yticks(np.arange(len(ordered_labels)))
        ax.set_yticklabels(ordered_labels, fontsize=8)
        ax.set_title(f"Cross-dataset ({scope_name}) transfer map — {pretty}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(pretty)
        out = out_dir / f"{scope_name}__heatmap_{val_col}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[OK] wrote {out}")

    agg_out = out_dir / f"{scope_name}__aggregated_mean_by_source_target.csv"
    agg.to_csv(agg_out, index=False)
    print(f"[OK] wrote {agg_out}")


def _process_one(csv_path: Path, scope_name: str, out_dir: Path) -> None:
    if not csv_path.exists():
        print(f"[WARN] missing input CSV for scope={scope_name}: {csv_path}")
        return
    df = pd.read_csv(csv_path, low_memory=False)
    if len(df) == 0:
        print(f"[WARN] empty input CSV for scope={scope_name}: {csv_path}")
        return
    dfx = _prep(df)
    _plot_target_boxplots(dfx, scope_name, out_dir)
    _plot_transfer_heatmaps(dfx, scope_name, out_dir)


def main() -> None:
    out_dir = OUTPUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _process_one(TEST_SPLIT_CSV.resolve(), "test_split", out_dir)
    _process_one(FULL_SCOPE_CSV.resolve(), "full_riva_herlev", out_dir)
    print(f"[DONE] Cross-dataset plots written to: {out_dir}")


if __name__ == "__main__":
    main()

