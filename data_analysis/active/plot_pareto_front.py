from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
ANALYSIS_DIR = Path("workspace/analysis")
EFF_CSV = ANALYSIS_DIR / "efficiency_profile" / "per_checkpoint_efficiency.csv"
TEST_CSV = ANALYSIS_DIR / "test_eval_results" / "per_checkpoint_test_metrics.csv"
OUT_DIR = ANALYSIS_DIR / "efficiency_profile" / "plots"

# Objective:
# - maximize accuracy / F1
# - minimize latency / params / MACs / FLOPs
ACC_COL = "test_acc"
F1_COL = "test_f1"
LAT_COL = "latency_mean_ms_last_k"
PARAMS_COL = "params_count"
MACS_COL = "macs_count"
FLOPS_COL = "flops_count"
MEM_COL = "memory_mean_mb_last_k"


def _model_family(model_name: str) -> str:
    name = str(model_name).lower()
    transformer_tokens = ("vit", "former", "iformer", "levit", "fastvit", "mobilevit", "eat")
    if any(tok in name for tok in transformer_tokens):
        return "transformer"
    return "cnn"


def _pareto_front(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    maximize_y: bool,
) -> pd.DataFrame:
    """
    Pareto front for 2D objective:
      - minimize x
      - maximize y (or minimize y if maximize_y=False)
    """
    d = df[[x_col, y_col]].copy()
    if not maximize_y:
        d[y_col] = -d[y_col]

    arr = d.to_numpy(dtype=float)
    keep = np.ones(len(arr), dtype=bool)
    for i in range(len(arr)):
        if not keep[i]:
            continue
        xi, yi = arr[i, 0], arr[i, 1]
        dominates_i = (
            (arr[:, 0] <= xi) &
            (arr[:, 1] >= yi) &
            ((arr[:, 0] < xi) | (arr[:, 1] > yi))
        )
        # if any point dominates i, i is not Pareto-optimal
        if dominates_i.any():
            keep[i] = False

    return df.loc[keep].copy()


def _scatter_with_front(
    df: pd.DataFrame,
    front: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    color_map = {"cnn": "#1f77b4", "transformer": "#d62728"}
    for fam in ("cnn", "transformer"):
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=22,
            alpha=0.65,
            c=color_map[fam],
            label=fam,
        )

    if not front.empty:
        f = front.sort_values(x_col, ascending=True)
        ax.scatter(
            f[x_col],
            f[y_col],
            s=56,
            marker="o",
            edgecolor="black",
            linewidth=0.9,
            c="#2ca02c",
            label="pareto_front",
            zorder=5,
        )
        ax.plot(
            f[x_col],
            f[y_col],
            linestyle="--",
            linewidth=1.1,
            c="#2ca02c",
            alpha=0.9,
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if x_col in {PARAMS_COL, MACS_COL, FLOPS_COL}:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    eff_csv = EFF_CSV.resolve()
    test_csv = TEST_CSV.resolve()
    if not eff_csv.exists():
        raise FileNotFoundError(f"Missing efficiency CSV: {eff_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test metrics CSV: {test_csv}")

    eff = pd.read_csv(eff_csv, low_memory=False)
    tst = pd.read_csv(test_csv, low_memory=False)

    need_eff = {"cfg_id", "dataset", "model", LAT_COL, PARAMS_COL, MACS_COL, FLOPS_COL, MEM_COL}
    need_tst = {"cfg_id", "dataset", "model", ACC_COL, F1_COL}
    miss_eff = sorted(need_eff - set(eff.columns))
    miss_tst = sorted(need_tst - set(tst.columns))
    if miss_eff:
        raise ValueError(f"Efficiency CSV missing columns: {miss_eff}")
    if miss_tst:
        raise ValueError(f"Test CSV missing columns: {miss_tst}")

    df = eff.merge(
        tst[["cfg_id", "dataset", "model", ACC_COL, F1_COL]],
        on=["cfg_id", "dataset", "model"],
        how="inner",
    )
    if df.empty:
        raise RuntimeError("No joined rows between efficiency and test metrics.")

    df["family"] = df["model"].map(_model_family)
    for col in [LAT_COL, PARAMS_COL, MACS_COL, FLOPS_COL, MEM_COL, ACC_COL, F1_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[LAT_COL, PARAMS_COL, MACS_COL, FLOPS_COL, MEM_COL, ACC_COL, F1_COL]).copy()

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_metrics = [
        (LAT_COL, "latency"),
        (PARAMS_COL, "params"),
        (MACS_COL, "macs"),
        (MEM_COL, "memory"),
    ]
    y_metrics = [
        (ACC_COL, "acc"),
        (F1_COL, "f1"),
    ]

    # Global fronts
    global_fronts: list[pd.DataFrame] = []
    for x_col, x_name in x_metrics:
        for y_col, y_name in y_metrics:
            front = _pareto_front(df, x_col=x_col, y_col=y_col, maximize_y=True)
            _scatter_with_front(
                df,
                front,
                x_col=x_col,
                y_col=y_col,
                title=f"Pareto Front (Global): {x_name} vs {y_name}",
                out_path=out_dir / f"pareto_global_{x_name}_vs_{y_name}.png",
            )
            global_fronts.append(front.assign(front_scope=f"global_{x_name}_vs_{y_name}"))

    # Dataset-specific fronts
    dataset_fronts: list[pd.DataFrame] = []
    for dataset in sorted(df["dataset"].astype(str).unique().tolist()):
        d = df[df["dataset"].astype(str) == dataset].copy()
        if d.empty:
            continue
        for x_col, x_name in x_metrics:
            for y_col, y_name in y_metrics:
                front = _pareto_front(d, x_col=x_col, y_col=y_col, maximize_y=True)
                _scatter_with_front(
                    d,
                    front,
                    x_col=x_col,
                    y_col=y_col,
                    title=f"Pareto Front ({dataset}): {x_name} vs {y_name}",
                    out_path=out_dir / f"pareto_{dataset}_{x_name}_vs_{y_name}.png",
                )
                dataset_fronts.append(front.assign(front_scope=f"{dataset}_{x_name}_vs_{y_name}"))

    # Save front membership CSVs
    if global_fronts:
        pd.concat(global_fronts, ignore_index=True).to_csv(
            out_dir / "pareto_front_global_all_scopes.csv",
            index=False,
        )
    if dataset_fronts:
        pd.concat(dataset_fronts, ignore_index=True).to_csv(
            out_dir / "pareto_front_by_dataset_all_scopes.csv",
            index=False,
        )

    print(f"[DONE] Pareto outputs written to: {out_dir}")


if __name__ == "__main__":
    main()

