from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
ANALYSIS_DIR = Path("workspace/analysis")
EFF_CSV = ANALYSIS_DIR / "efficiency_profile" / "per_checkpoint_efficiency.csv"
TEST_CSV = ANALYSIS_DIR / "test_eval_results" / "per_checkpoint_test_metrics.csv"
OUT_DIR = ANALYSIS_DIR / "efficiency_profile" / "plots_efficiency_vs_performance"

# x metrics from efficiency CSV
X_COLS = [
    ("params_count", "Parameter Count", 1e6, "M params"),
    ("macs_count", "MACs", 1e9, "G MACs"),
    ("memory_mean_mb_last_k", "Memory Footprint", 1.0, "MB"),
]

# y metrics from test CSV
Y_COLS = [
    ("test_acc", "Accuracy"),
    ("test_f1", "F1-score"),
]


def _model_family(model_name: str) -> str:
    name = str(model_name).lower()
    transformer_tokens = ("vit", "former", "iformer", "levit", "fastvit", "mobilevit", "eat")
    if any(tok in name for tok in transformer_tokens):
        return "transformer"
    return "cnn"


def _merge_inputs(eff: pd.DataFrame, tst: pd.DataFrame) -> pd.DataFrame:
    keys = ["cfg_id", "dataset", "model", "origin", "fold", "best_epoch"]
    for k in keys:
        if k not in eff.columns or k not in tst.columns:
            raise ValueError(f"Missing join key {k!r} in input CSVs.")
    df = eff.merge(
        tst[keys + [y for y, _ in Y_COLS]],
        on=keys,
        how="inner",
    )
    if df.empty:
        raise RuntimeError("No rows after joining efficiency and test CSVs.")
    df["family"] = df["model"].map(_model_family)
    return df


def _scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    x_title: str,
    x_div: float,
    x_unit: str,
    y_col: str,
    y_title: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    color_map = {"cnn": "#1f77b4", "transformer": "#d62728"}

    x_plot = pd.to_numeric(df[x_col], errors="coerce") / x_div
    y_plot = pd.to_numeric(df[y_col], errors="coerce")
    plot_df = df.assign(_x=x_plot, _y=y_plot).dropna(subset=["_x", "_y"])

    for fam in ("cnn", "transformer"):
        d = plot_df[plot_df["family"] == fam]
        if d.empty:
            continue
        ax.scatter(
            d["_x"],
            d["_y"],
            s=26,
            alpha=0.7,
            c=color_map[fam],
            label=fam,
        )

    # Mark key extremes with model names:
    # - best performer (highest y)
    # - leftmost (lowest x)
    # - rightmost (highest x)
    # - worst performer (lowest y)
    if not plot_df.empty:
        key_points = [
            ("best", int(plot_df["_y"].idxmax())),
            ("leftmost", int(plot_df["_x"].idxmin())),
            ("rightmost", int(plot_df["_x"].idxmax())),
            ("worst", int(plot_df["_y"].idxmin())),
        ]
        # Keep first occurrence order while removing duplicate indices.
        seen = set()
        unique_points: list[tuple[str, int]] = []
        for tag, idx in key_points:
            if idx not in seen:
                seen.add(idx)
                unique_points.append((tag, idx))

        offsets = [(-12, 10), (8, 12), (8, -12), (-12, -12)]
        for i, (tag, idx) in enumerate(unique_points):
            r = plot_df.loc[idx]
            label = f"{tag}: {r['model']}"
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                label,
                xy=(float(r["_x"]), float(r["_y"])),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75},
                arrowprops={"arrowstyle": "->", "lw": 0.8, "alpha": 0.7},
            )

    ax.set_title(title)
    ax.set_xlabel(f"{x_title} ({x_unit})")
    ax.set_ylabel(y_title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def _plot_scope(df: pd.DataFrame, scope_name: str, out_dir: Path) -> None:
    for x_col, x_title, x_div, x_unit in X_COLS:
        if x_col not in df.columns:
            print(f"[WARN] scope={scope_name}: missing {x_col}, skipping")
            continue
        for y_col, y_title in Y_COLS:
            if y_col not in df.columns:
                print(f"[WARN] scope={scope_name}: missing {y_col}, skipping")
                continue
            out = out_dir / f"{scope_name}__{x_col}_vs_{y_col}.png"
            _scatter(
                df,
                x_col=x_col,
                x_title=x_title,
                x_div=x_div,
                x_unit=x_unit,
                y_col=y_col,
                y_title=y_title,
                title=f"{scope_name}: {x_title} vs {y_title}",
                out_path=out,
            )


def main() -> None:
    eff_csv = EFF_CSV.resolve()
    tst_csv = TEST_CSV.resolve()
    if not eff_csv.exists():
        raise FileNotFoundError(f"Missing efficiency CSV: {eff_csv}")
    if not tst_csv.exists():
        raise FileNotFoundError(f"Missing test metrics CSV: {tst_csv}")

    eff = pd.read_csv(eff_csv, low_memory=False)
    tst = pd.read_csv(tst_csv, low_memory=False)
    df = _merge_inputs(eff, tst)

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global
    _plot_scope(df, "global", out_dir)

    # Per dataset
    for dataset in sorted(df["dataset"].astype(str).unique().tolist()):
        d = df[df["dataset"].astype(str) == dataset].copy()
        if d.empty:
            continue
        _plot_scope(d, f"dataset_{dataset}", out_dir)

    print(f"[DONE] Efficiency vs performance plots written to: {out_dir}")


if __name__ == "__main__":
    main()

