from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from data_analysis.active.lib.dataset_regime_utils import DATASET_DISPLAY, SOLO_DATASETS


MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/analysis/generalizability/plots/transfer_heatmaps")
METRIC = "target_f1"

# Consistent held-out column order (Herlev, Riva, SIPaKMeD).
TARGET_COLUMN_ORDER: tuple[str, ...] = ("herlev", "riva", "sipakmed")

# Reference panel width for mixed heatmap (inches); solo fonts scale to match relative size.
MIXED_PANEL_WIDTH_IN = 6.8
MIXED_FONT_CELL = 8.5
MIXED_FONT_TICK = 9.0
MIXED_FONT_TITLE = 12.0


def _held_out_column_labels(slugs: tuple[str, ...]) -> list[str]:
    return [DATASET_DISPLAY[ds] for ds in slugs]


def _held_out_targets_excluding(train_dataset: str) -> tuple[str, ...]:
    return tuple(ds for ds in TARGET_COLUMN_ORDER if ds != train_dataset)


def _annotate_heatmap_cells(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".2f",
    fontsize: float = 8.5,
    light_threshold: float = 0.52,
) -> None:
    n_rows, n_cols = values.shape
    span = vmax - vmin
    for row in range(n_rows):
        for col in range(n_cols):
            val = float(values[row, col])
            norm = (val - vmin) / span if span > 0 else 0.5
            text_color = "white" if norm < light_threshold else "black"
            ax.text(
                col,
                row,
                format(val, fmt),
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
            )


def _build_model_target_matrix(
    df: pd.DataFrame,
    *,
    regime: str,
    train_dataset: str | None = None,
) -> pd.DataFrame:
    """Rows = models, columns = held-out targets (display names), sorted by mean F1."""
    sub = df[df["source_dataset_regime"].astype(str) == regime].copy()
    if train_dataset is not None:
        sub = sub[sub["source_dataset"].astype(str) == train_dataset].copy()
    if sub.empty:
        raise ValueError(f"No rows for regime={regime!r}, train_dataset={train_dataset!r}.")

    held_out = _held_out_targets_excluding(train_dataset) if train_dataset else TARGET_COLUMN_ORDER

    agg = (
        sub.groupby(["source_model", "target_dataset"], as_index=False)[METRIC]
        .mean()
        .rename(columns={METRIC: "f1"})
    )
    pivot = agg.pivot(index="source_model", columns="target_dataset", values="f1")
    missing_cols = [c for c in held_out if c not in pivot.columns]
    if missing_cols:
        raise ValueError(f"Heatmap missing held-out targets {missing_cols} (train={train_dataset!r}).")

    pivot = pivot.reindex(columns=list(held_out))
    pivot.columns = _held_out_column_labels(held_out)

    row_mean = pivot.mean(axis=1, skipna=True)
    pivot = pivot.loc[row_mean.sort_values(ascending=False).index]

    if pivot.isna().any().any():
        missing = pivot[pivot.isna().any(axis=1)]
        raise ValueError(f"Heatmap has missing cells for models:\n{missing.index.tolist()}")

    return pivot


def _heatmap_font_sizes(*, panel_width_in: float) -> dict[str, float]:
    """Scale fonts so solo panels match mixed heatmap text-to-cell proportions."""
    scale = MIXED_PANEL_WIDTH_IN / panel_width_in
    return {
        "cell": MIXED_FONT_CELL * scale,
        "tick": MIXED_FONT_TICK * scale,
        "title": MIXED_FONT_TITLE * scale,
        "suptitle": MIXED_FONT_TITLE * scale,
    }


def _figure_size(n_rows: int, n_cols: int) -> tuple[float, float]:
    """Match mixed heatmap proportions; narrower width when fewer target columns."""
    row_h = 0.38
    base_w = 6.8
    width = max(4.6, base_w * (n_cols / len(TARGET_COLUMN_ORDER)))
    height = max(5.0, row_h * n_rows)
    return width, height


def _plot_model_target_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    out_stem: str,
    out_dir: Path,
    cell_fontsize: float = 8.5,
) -> None:
    values = pivot.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    fig_w, fig_h = _figure_size(n_rows, n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    _annotate_heatmap_cells(ax, values, vmin=0.0, vmax=1.0, fmt=".2f", fontsize=cell_fontsize)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Held-out target dataset")
    ax.set_ylabel("Model")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1-score")

    out_png = out_dir / f"{out_stem}.png"
    out_csv = out_dir / f"{out_stem}.csv"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    pivot.to_csv(out_csv)
    print(f"[OK] wrote {out_png}")
    print(f"[OK] wrote {out_csv}")


def _plot_mixed_transfer_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = _build_model_target_matrix(df, regime="mixed")
    _plot_model_target_heatmap(
        pivot,
        title="Mixed-source cross-dataset generalization",
        out_stem="transfer_heatmap_mixed",
        out_dir=out_dir,
    )


def _unified_model_order(pivots: list[pd.DataFrame]) -> list[str]:
    """One row order for all solo panels: mean OOD F1 across every held-out cell."""
    models = pivots[0].index.tolist()
    mean_scores: dict[str, float] = {}
    for model in models:
        vals: list[float] = []
        for pivot in pivots:
            vals.extend(float(v) for v in pivot.loc[model].tolist())
        mean_scores[model] = float(np.mean(vals))
    return sorted(mean_scores, key=mean_scores.get, reverse=True)  # type: ignore[arg-type]


def _plot_solo_composite_heatmap(pivots: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Three solo heatmaps side-by-side with a single shared colorbar."""
    train_order = [ds for ds in TARGET_COLUMN_ORDER if ds in pivots]
    ordered_pivots = [pivots[ds] for ds in train_order]
    model_order = _unified_model_order(ordered_pivots)
    ordered_pivots = [p.loc[model_order] for p in ordered_pivots]

    n_rows = len(model_order)
    n_panels = len(ordered_pivots)
    panel_w, panel_h = _figure_size(n_rows, 2)
    fonts = _heatmap_font_sizes(panel_width_in=panel_w)
    cbar_w = 0.55
    fig = plt.figure(figsize=(panel_w * n_panels + cbar_w, panel_h))
    gs = GridSpec(
        1,
        n_panels + 1,
        figure=fig,
        width_ratios=[1] * n_panels + [0.04],
        wspace=0.22,
    )

    vmin, vmax = 0.0, 1.0
    cmap = "viridis"
    im = None
    for col_idx, (train_dataset, pivot) in enumerate(zip(train_order, ordered_pivots, strict=True)):
        ax = fig.add_subplot(gs[0, col_idx])
        values = pivot.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        _annotate_heatmap_cells(
            ax, values, vmin=vmin, vmax=vmax, fmt=".2f", fontsize=fonts["cell"]
        )
        train_name = DATASET_DISPLAY[train_dataset]
        ax.set_title(f"Train: {train_name}", fontsize=fonts["title"], pad=8)
        ax.set_xlabel("Held-out target", fontsize=fonts["tick"])
        ax.set_xticks(range(values.shape[1]))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=0, fontsize=fonts["tick"])
        ax.set_yticks(range(n_rows))
        if col_idx == 0:
            ax.set_ylabel("Model", fontsize=fonts["tick"])
            ax.set_yticklabels(pivot.index.tolist(), fontsize=fonts["tick"])
        else:
            ax.set_yticklabels([])

    cax = fig.add_subplot(gs[0, n_panels])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("F1-score", fontsize=fonts["tick"])
    cbar.ax.tick_params(labelsize=fonts["tick"])
    fig.suptitle(
        "Single-source cross-dataset generalization",
        fontsize=fonts["suptitle"],
        y=1.02,
    )

    out_png = out_dir / "transfer_heatmap_solo.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_png}")

    for train_dataset, pivot in zip(train_order, ordered_pivots, strict=True):
        out_csv = out_dir / f"transfer_heatmap_solo_{train_dataset}.csv"
        pivot.to_csv(out_csv)
        print(f"[OK] wrote {out_csv}")


def _plot_solo_transfer_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    pivots = {
        train_dataset: _build_model_target_matrix(df, regime="solo", train_dataset=train_dataset)
        for train_dataset in TARGET_COLUMN_ORDER
        if train_dataset in SOLO_DATASETS
    }
    _plot_solo_composite_heatmap(pivots, out_dir)


def main() -> None:
    master_path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master CSV: {master_path}")
    df = pd.read_csv(master_path, low_memory=False)
    required = {
        "source_dataset_regime",
        "source_dataset",
        "source_model",
        "target_dataset",
        METRIC,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df = df.dropna(subset=[METRIC]).copy()

    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_mixed_transfer_heatmaps(df, out_dir)
    _plot_solo_transfer_heatmaps(df, out_dir)


if __name__ == "__main__":
    main()
