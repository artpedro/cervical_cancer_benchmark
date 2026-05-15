from __future__ import annotations

"""
Plot aggregated train vs validation loss curves across datasets.
Creates a single figure per model, containing subplots for each dataset.
Plots individual fold curves with low alpha and a mean curve with higher alpha.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_analysis.active.dataset_regime_utils import is_mixed_dataset
from training.io_utils import slugify
from data_analysis.active.plot_metrics_loss_curves import (
    _parse_epoch_log_path,
    _require_columns,
    _select_epoch_log_csvs,
)

WORKSPACE = Path("workspace")
DATASET_GROUP_MODE = "all"  # all | solo_only | mixed_only

SPLIT_STYLE: dict[str, tuple[str, str]] = {
    "train": ("Train", "#1f77b4"),
    "val": ("Validation", "#ff7f0e"),
    "test": ("Test", "#2ca02c"),
}

def main() -> None:
    ws = (WORKSPACE if WORKSPACE.is_absolute() else _REPO_ROOT / WORKSPACE).resolve()
    if not ws.is_dir():
        raise FileNotFoundError(f"Workspace not found: {ws}")

    paths = _select_epoch_log_csvs(ws)
    if not paths:
        raise RuntimeError(f"No epoch_logs_*.csv under {ws}/metrics*/")

    dfs = []
    for csv_path in paths:
        parsed = _parse_epoch_log_path(csv_path, ws)
        if parsed is None:
            continue
        metrics_root, _dataset, _balance_mode, _run_ts = parsed
        df = pd.read_csv(csv_path, low_memory=False)
        _require_columns(df, ["dataset", "model", "fold", "epoch", "split", "loss"])
        df = df.assign(metrics_root=metrics_root)
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No parseable epoch_logs_*.csv files found.")
    
    full_df = pd.concat(dfs, ignore_index=True)
    if DATASET_GROUP_MODE == "solo_only":
        full_df = full_df[~full_df["dataset"].astype(str).map(is_mixed_dataset)].copy()
    elif DATASET_GROUP_MODE == "mixed_only":
        full_df = full_df[full_df["dataset"].astype(str).map(is_mixed_dataset)].copy()
    models = sorted(full_df["model"].astype(str).unique())
    
    out_dir = ws / "plots" / "loss_curves_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    written = 0
    for model in models:
        m_df = full_df[full_df["model"].astype(str) == model]
        # Only keep datasets that this model has data for
        m_datasets = sorted(m_df["dataset"].astype(str).unique())
        if not m_datasets:
            continue
            
        n_datasets = len(m_datasets)
        # A 1 x N figure (horizontal layout)
        fig_w = max(10.0, 4.0 * n_datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(fig_w, 4.0), squeeze=False, constrained_layout=True)
        
        for ax, dataset in zip(axes[0, :], m_datasets, strict=False):
            d_df = m_df[m_df["dataset"].astype(str) == dataset]
            
            # 1) Plot low alpha individual folds
            folds = sorted(d_df["fold"].dropna().unique())
            for fold in folds:
                f_df = d_df[d_df["fold"] == fold]
                for split_key, (_label, color) in SPLIT_STYLE.items():
                    s_df = f_df[f_df["split"].astype(str) == split_key].sort_values("epoch")
                    if not s_df.empty:
                        # plot each fold without label to avoid legend clutter
                        ax.plot(
                            s_df["epoch"].to_numpy(),
                            s_df["loss"].to_numpy(),
                            color=color,
                            alpha=0.15,
                            linewidth=1.0
                        )
            
            # 2) Plot mean curve
            for split_key, (label, color) in SPLIT_STYLE.items():
                s_df = d_df[d_df["split"].astype(str) == split_key]
                if s_df.empty:
                    continue
                mean_df = s_df.groupby("epoch")["loss"].mean().reset_index().sort_values("epoch")
                ax.plot(
                    mean_df["epoch"].to_numpy(),
                    mean_df["loss"].to_numpy(),
                    color=color,
                    alpha=0.9,
                    linewidth=2.0,
                    label=f"{label} (Mean)"
                )
                
            ax.set_title(f"{dataset}", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(fontsize=8, loc="upper right")
            
            # Robust scaling (same logic as single plot)
            valid_losses = d_df["loss"].dropna()
            if not valid_losses.empty:
                q1 = np.percentile(valid_losses, 25)
                q3 = np.percentile(valid_losses, 75)
                iqr = q3 - q1
                y_upper = q3 + 5.0 * iqr
                y_upper = max(y_upper, np.percentile(valid_losses, 90) * 1.5)
                y_max = min(valid_losses.max(), y_upper)
                y_max = max(y_max * 1.05, 1e-3)
                ax.set_ylim(bottom=-0.02 * y_max, top=y_max)
                
        axes[0, 0].set_ylabel("Loss")
        fig.suptitle(f"Aggregated Loss Curves: {model}", fontsize=13, fontweight="bold")
        
        out_path = out_dir / f"loss_aggregated__{slugify(model)}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        written += 1
        print(f"[OK] Wrote {out_path.relative_to(ws)}")

    print(f"[DONE] Wrote {written} figure(s) under workspace/plots/loss_curves_aggregated/")

if __name__ == '__main__':
    main()
