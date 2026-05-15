from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/analysis/generalizability/plots/transfer_heatmaps")
METRIC = "target_f1"
MAX_SOURCES_PER_REGIME = 35


def _plot_regime_heatmap(df: pd.DataFrame, regime: str, out_dir: Path) -> None:
    sub = df[df["source_dataset_regime"].astype(str) == regime].copy()
    if sub.empty:
        print(f"[WARN] no rows for regime={regime}")
        return

    agg = (
        sub.groupby(["source_label", "target_dataset"], as_index=False)[METRIC]
        .mean()
        .rename(columns={METRIC: "metric_mean"})
    )
    source_rank = (
        agg.groupby("source_label", as_index=False)["metric_mean"]
        .mean()
        .sort_values("metric_mean", ascending=False)
    )
    keep_sources = source_rank["source_label"].head(MAX_SOURCES_PER_REGIME).tolist()
    agg = agg[agg["source_label"].isin(keep_sources)].copy()

    pivot = agg.pivot(index="source_label", columns="target_dataset", values="metric_mean")
    pivot = pivot.loc[keep_sources]
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig_h = max(4.0, 0.24 * len(pivot))
    fig, ax = plt.subplots(figsize=(7.8, fig_h), constrained_layout=True)
    im = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"Transfer Heatmap ({regime})")
    ax.set_xlabel("Target Dataset")
    ax.set_ylabel("Source Dataset + Model")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean target F1")

    out_png = out_dir / f"transfer_heatmap_{regime}.png"
    out_csv = out_dir / f"transfer_heatmap_{regime}.csv"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    pivot.to_csv(out_csv)
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_csv}")


def main() -> None:
    master_path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master CSV: {master_path}")
    df = pd.read_csv(master_path, low_memory=False)
    required = {"source_dataset_regime", "source_label", "target_dataset", METRIC}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df = df.dropna(subset=[METRIC]).copy()

    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for regime in ("solo", "mixed"):
        _plot_regime_heatmap(df, regime, out_dir)


if __name__ == "__main__":
    main()
