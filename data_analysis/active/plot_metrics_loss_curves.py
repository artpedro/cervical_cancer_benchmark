from __future__ import annotations

"""
Plot train vs validation loss curves from training epoch logs under workspace/metrics*.

Reads CSVs matching: workspace/metrics*/<dataset>/<balance_mode>/<timestamp>/epoch_logs_*_*.csv
Writes PNGs to:     workspace/<metrics_root>/plots/loss_curves/<dataset>/

Validation split is the CV fold holdout (logged as "val" in pipeline); shown as
"Validation (fold holdout)" in the legend. If a "test" split exists, it is plotted too.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.io_utils import slugify

WORKSPACE = Path("workspace")
# If True, only the newest run (by timestamp folder name) per (metrics_root, dataset, balance_mode).
LATEST_RUN_ONLY = False

SPLIT_STYLE: dict[str, tuple[str, str]] = {
    "train": ("Train", "#1f77b4"),
    "val": ("Validation (fold holdout)", "#ff7f0e"),
    "test": ("Test", "#2ca02c"),
}


def _parse_epoch_log_path(csv_path: Path, workspace: Path) -> tuple[str, str, str, str] | None:
    try:
        rel = csv_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5:
        return None
    metrics_root, dataset, balance_mode, run_ts, _fname = (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        parts[4],
    )
    if not metrics_root.startswith("metrics"):
        return None
    return metrics_root, dataset, balance_mode, run_ts


def _select_epoch_log_csvs(workspace: Path) -> list[Path]:
    all_paths = sorted(workspace.glob("metrics*/**/epoch_logs_*.csv"))
    if not LATEST_RUN_ONLY:
        return all_paths
    best: dict[tuple[str, str, str], tuple[str, Path]] = {}
    for p in all_paths:
        parsed = _parse_epoch_log_path(p, workspace)
        if parsed is None:
            continue
        metrics_root, dataset, balance_mode, run_ts = parsed
        key = (metrics_root, dataset, balance_mode)
        prev = best.get(key)
        if prev is None or run_ts > prev[0]:
            best[key] = (run_ts, p)
    return sorted(path for _ts, path in best.values())


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in epoch log")


def _plot_one_model_one_csv(
    df: pd.DataFrame,
    *,
    model_name: str,
    dataset: str,
    balance_mode: str,
    run_ts: str,
    out_path: Path,
) -> None:
    sub = df.loc[df["model"].astype(str) == model_name].copy()
    if sub.empty:
        return
    folds = sorted(sub["fold"].dropna().unique().tolist())
    n = len(folds)
    if n == 0:
        return

    fig_w = max(10.0, 2.85 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.6), sharey=True, constrained_layout=True)
    ax_arr = np.atleast_1d(axes)

    for ax, fold in zip(ax_arr, folds, strict=True):
        d = sub.loc[sub["fold"] == fold].sort_values("epoch")
        for split_key in ("train", "val", "test"):
            if split_key not in SPLIT_STYLE:
                continue
            s = d.loc[d["split"].astype(str) == split_key]
            if s.empty:
                continue
            label, color = SPLIT_STYLE[split_key]
            ax.plot(
                s["epoch"].to_numpy(),
                s["loss"].to_numpy(),
                label=label,
                color=color,
                linewidth=1.35,
                alpha=0.9,
            )
        ax.set_xlabel("Epoch")
        ax.set_title(f"Fold {int(fold)}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, loc="upper right")

    valid_losses = sub["loss"].dropna()
    if not valid_losses.empty:
        q1 = np.percentile(valid_losses, 25)
        q3 = np.percentile(valid_losses, 75)
        iqr = q3 - q1
        # Set a robust upper limit to filter out huge outlier peaks
        y_upper = q3 + 5.0 * iqr
        # Ensure we don't clip too much of the valid early training loss
        y_upper = max(y_upper, np.percentile(valid_losses, 90) * 1.5)
        y_max = min(valid_losses.max(), y_upper)
        
        # Add a little padding and ensure it's strictly positive
        y_max = max(y_max * 1.05, 1e-3)
        ax_arr[0].set_ylim(bottom=-0.02 * y_max, top=y_max)

    ax_arr[0].set_ylabel("Loss")
    fig.suptitle(
        f"{dataset} — {model_name}\n{balance_mode} · run {run_ts}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ws = (WORKSPACE if WORKSPACE.is_absolute() else _REPO_ROOT / WORKSPACE).resolve()
    if not ws.is_dir():
        raise FileNotFoundError(f"Workspace not found: {ws}")

    paths = _select_epoch_log_csvs(ws)
    if not paths:
        raise RuntimeError(f"No epoch_logs_*.csv under {ws}/metrics*/")

    written = 0
    for csv_path in paths:
        parsed = _parse_epoch_log_path(csv_path, ws)
        if parsed is None:
            continue
        metrics_root, dataset, balance_mode, run_ts = parsed
        df = pd.read_csv(csv_path, low_memory=False)
        _require_columns(df, ["dataset", "model", "fold", "epoch", "split", "loss"])

        out_dir = ws / metrics_root / "plots" / "loss_curves" / slugify(dataset)
        for model_name in sorted(df["model"].astype(str).unique().tolist()):
            slug = slugify(model_name)
            if not LATEST_RUN_ONLY:
                fname = f"loss__{slugify(dataset)}__{slug}__{run_ts}.png"
            else:
                fname = f"loss__{slugify(dataset)}__{slug}.png"
            out_path = out_dir / fname
            _plot_one_model_one_csv(
                df,
                model_name=model_name,
                dataset=dataset,
                balance_mode=balance_mode,
                run_ts=run_ts,
                out_path=out_path,
            )
            written += 1
            print(f"[OK] {out_path.relative_to(ws)}")

    print(f"[DONE] Wrote {written} figure(s) under workspace/*/plots/loss_curves/")


if __name__ == "__main__":
    main()
