from __future__ import annotations

"""
Per-model macro F1 over the three in-dataset test benchmarks, vs mean latency.

For each model:
  - mean test F1 across CV folds on herlev (train+test on herlev)
  - same for sipakmed and riva
  - macro_f1 = (f1_herlev + f1_sipakmed + f1_riva) / 3

Per-checkpoint latency_mean_ms_last_k is the median of many timed forward passes
(profile_checkpoint_efficiency); the macro plot then averages that value across
herlev/sipakmed/riva checkpoints for each model (15 values when all folds exist).
Basis (batch vs per-image) is taken from column latency_ms_basis in the efficiency CSV
(profile_checkpoint_efficiency.LATENCY_MS_REPORT).

Parameter count is the mean of params_count over those same rows (identical across
folds for a given model; mean is for robustness).

Bubble area is proportional to parameter count. Colors: blue = CNN, red =
transformer (state in caption if needed). A small upper-left legend maps bubble
area to mean parameter count.

Model names are placed with adjustText so labels avoid overlap and stay inside
the axes (requires the adjusttext package).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D

from data_analysis.active.dataset_regime_utils import SOLO_DATASETS, is_mixed_dataset


ANALYSIS_DIR = Path("workspace/analysis")
TEST_CSV = ANALYSIS_DIR / "test_eval_results" / "per_checkpoint_test_metrics.csv"
EFF_CSV = ANALYSIS_DIR / "efficiency_profile" / "per_checkpoint_efficiency.csv"
OUT_DIR = ANALYSIS_DIR / "efficiency_profile" / "plots"
OUT_CSV = ANALYSIS_DIR / "efficiency_profile" / "macro_f1_vs_latency_per_model.csv"

DATASETS: tuple[str, ...] | None = SOLO_DATASETS
INCLUDE_MIXED_DATASETS = False
F1_COL = "test_f1"
LAT_COL = "latency_mean_ms_last_k"
PARAMS_COL = "params_count"

COLOR_CNN = "#3a76af"
COLOR_TRANSFORMER = "#c44e52"

MODEL_SHORT_NAMES: dict[str, str] = {
    "EfficientNet B0": "EN-B0",
    "EfficientNet B1": "EN-B1",
    "EfficientNet B2": "EN-B2",
    "MobileNet V2": "MBNetV2",
    "MobileNet V4": "MBNetV4",
    "EfficientFormerV2 S0": "EffF-S0",
    "EfficientFormerV2 S1": "EffF-S1",
    "MobileViT v2 100": "MViTv2",
    "FastViT T8": "FastViT-T8",
    "iformer_m": "iFormer-M",
}

# Academic figure: single-column ~89 mm ≈ 3.5 in; 4:3 aspect.
FIG_W, FIG_H = 3.54, 2.80
DPI = 300


def _latency_xlabel(eff: pd.DataFrame) -> str:
    if "latency_ms_basis" not in eff.columns:
        return "Latency (ms / batch)"
    modes = eff["latency_ms_basis"].dropna().astype(str).str.lower()
    if modes.empty:
        return "Latency (ms / batch)"
    mode = modes.mode()
    basis = str(mode.iloc[0]) if len(mode) else str(modes.iloc[0])
    return "Latency (ms / image)" if basis == "image" else "Latency (ms / batch)"


def _model_family(model_name: str) -> str:
    name = str(model_name).lower()
    transformer_tokens = ("vit", "former", "iformer", "levit", "fastvit", "mobilevit", "eat")
    if any(tok in name for tok in transformer_tokens):
        return "transformer"
    return "cnn"


def _macro_f1_table(tst: pd.DataFrame) -> pd.DataFrame:
    datasets = list(DATASETS) if DATASETS else sorted(tst["dataset"].dropna().astype(str).unique().tolist())
    if not INCLUDE_MIXED_DATASETS:
        datasets = [d for d in datasets if not is_mixed_dataset(d)]
    sub = tst[tst["dataset"].astype(str).isin(datasets)].copy()
    sub[F1_COL] = pd.to_numeric(sub[F1_COL], errors="coerce")
    sub = sub.dropna(subset=[F1_COL])
    g = sub.groupby(["model", "dataset"], as_index=False)[F1_COL].mean()
    wide = g.pivot(index="model", columns="dataset", values=F1_COL)
    for ds in datasets:
        if ds not in wide.columns:
            wide[ds] = np.nan
    wide = wide[list(datasets)]
    complete = wide.dropna(how="any")
    ds_cols = list(datasets)
    complete["macro_f1"] = complete[ds_cols].mean(axis=1)
    return complete.reset_index()


def _mean_latency_and_params_per_model(eff: pd.DataFrame) -> pd.DataFrame:
    datasets = list(DATASETS) if DATASETS else sorted(eff["dataset"].dropna().astype(str).unique().tolist())
    if not INCLUDE_MIXED_DATASETS:
        datasets = [d for d in datasets if not is_mixed_dataset(d)]
    sub = eff[eff["dataset"].astype(str).isin(datasets)].copy()
    sub[LAT_COL] = pd.to_numeric(sub[LAT_COL], errors="coerce")
    sub[PARAMS_COL] = pd.to_numeric(sub[PARAMS_COL], errors="coerce")
    sub = sub.dropna(subset=[LAT_COL, PARAMS_COL])
    g = sub.groupby("model", as_index=False).agg(
        latency_mean_ms_all_checkpoints=(LAT_COL, "mean"),
        params_count_mean=(PARAMS_COL, "mean"),
    )
    return g


def _apply_macro_latency_axis_ticks(
    ax,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_pad: float,
    y_pad: float,
) -> None:
    """Major ticks for latency (x) and macro F1 (y); works for ms/image and ms/batch."""
    x_lim_span = (x_max + x_pad) - (x_min - x_pad)
    y_lim_span = (y_max + y_pad) - (y_min - y_pad)

    ax.xaxis.set_major_locator(
        mticker.MaxNLocator(nbins=10, min_n_ticks=7, prune=None)
    )
    if y_lim_span <= 0.14:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    else:
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=10, min_n_ticks=7, prune=None)
        )

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    if x_lim_span > 40:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    else:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


def main() -> None:
    test_path = TEST_CSV.resolve()
    eff_path = EFF_CSV.resolve()
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test metrics CSV: {test_path}")
    if not eff_path.exists():
        raise FileNotFoundError(f"Missing efficiency CSV: {eff_path}")

    tst = pd.read_csv(test_path, low_memory=False)
    eff = pd.read_csv(eff_path, low_memory=False)
    for col in ("model", "dataset", F1_COL):
        if col not in tst.columns:
            raise ValueError(f"Test CSV missing column: {col}")
    for col in (LAT_COL, PARAMS_COL):
        if col not in eff.columns:
            raise ValueError(f"Efficiency CSV missing column: {col}")

    f1_tbl = _macro_f1_table(tst)
    eff_tbl = _mean_latency_and_params_per_model(eff)
    df = f1_tbl.merge(eff_tbl, on="model", how="inner")
    if df.empty:
        raise RuntimeError("No models after merging macro F1 and latency.")

    df = df.reset_index(drop=True)
    df["family"] = df["model"].map(_model_family)
    x_col = "latency_mean_ms_all_checkpoints"
    y_col = "macro_f1"
    p_col = "params_count_mean"

    # Matplotlib scatter `s` is marker area in points². Scale linearly so area ∝ params.
    p = df[p_col].to_numpy(dtype=float)
    p_lo, p_hi = float(np.min(p)), float(np.max(p))
    span = p_hi - p_lo
    s_lo, s_hi = 40.0, 320.0
    if span <= 0:
        s = np.full(len(df), (s_lo + s_hi) / 2.0)
    else:
        t = (p - p_lo) / span
        s = s_lo + t * (s_hi - s_lo)

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_CSV.resolve()
    df.sort_values(y_col, ascending=False).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }

    plot_path = out_dir / "macro_f1_vs_latency.png"
    plot_pdf = out_dir / "macro_f1_vs_latency.pdf"

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
        ax.set_axisbelow(True)

        for fam, rgb in [("cnn", COLOR_CNN), ("transformer", COLOR_TRANSFORMER)]:
            m = df["family"] == fam
            if not m.any():
                continue
            idx = np.flatnonzero(m.to_numpy())
            ax.scatter(
                df.loc[m, x_col],
                df.loc[m, y_col],
                s=s[idx],
                alpha=0.80,
                c=rgb,
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )

        x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
        y_min, y_max = float(df[y_col].min()), float(df[y_col].max())
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_pad = max(x_span * 0.14, 0.4)
        y_pad = max(y_span * 0.14, 0.005)
        ax.set_xlim(x_min - x_pad, x_max + x_pad * 1.6)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        _apply_macro_latency_axis_ticks(
            ax,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            x_pad=x_pad, y_pad=y_pad,
        )

        ax.set_xlabel(_latency_xlabel(eff))
        ax.set_ylabel("Macro F1")

        ax.grid(True, which="major", axis="both",
                linestyle="--", linewidth=0.35, color="#c8c8c8", alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.25")
        ax.spines["bottom"].set_color("0.25")

        # --- Parameter-count size legend ---
        # Use same s_lo / s_hi mapping as scatter so legend circles match data.
        if span > 0:
            refs = [p_lo, (p_lo + p_hi) / 2.0, p_hi]
        else:
            refs = [p_lo, p_lo, p_lo]
        ref_s_vals: list[float] = []
        for pv in refs:
            if span <= 0:
                ref_s_vals.append((s_lo + s_hi) / 2.0)
            else:
                t_val = (pv - p_lo) / span
                ref_s_vals.append(s_lo + t_val * (s_hi - s_lo))

        def _params_legend_label(n: float) -> str:
            if n >= 1e6:
                return f"{n / 1e6:.1f} M"
            return f"{n / 1e3:.0f} k"

        # scatter s = area in points²; Line2D markersize = diameter in points.
        # diameter = 2 * sqrt(area / pi).
        size_handles = [
            Line2D(
                [0], [0], linestyle="none", marker="o",
                markersize=2.0 * np.sqrt(sv / np.pi),
                markerfacecolor="0.50",
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=_params_legend_label(rv),
            )
            for sv, rv in zip(ref_s_vals, refs)
        ]
        leg = ax.legend(
            handles=size_handles,
            loc="upper left",
            title="Parameters",
            frameon=True,
            edgecolor="0.78",
            facecolor="white",
            framealpha=0.92,
            fontsize=6,
            title_fontsize=6.5,
            borderpad=0.45,
            labelspacing=0.85,
            handletextpad=0.6,
        )
        leg.get_frame().set_linewidth(0.45)
        leg.set_zorder(10)

        # --- Short model-name labels via adjustText ---
        xs = df[x_col].to_numpy(dtype=float)
        ys = df[y_col].to_numpy(dtype=float)
        labels = [MODEL_SHORT_NAMES.get(str(n), str(n)) for n in df["model"]]

        texts = [
            ax.text(
                float(xs[i]), float(ys[i]),
                labels[i],
                fontsize=6, fontweight="medium",
                color="0.12",
                ha="center", va="center",
                zorder=5,
            )
            for i in range(len(df))
        ]

        fig.canvas.draw()
        adjust_text(
            texts,
            x=xs, y=ys,
            ax=ax,
            objects=[leg],
            arrowprops=None,
            expand=(2.0, 2.4),
            force_text=(0.25, 0.45),
            force_static=(55, 35),
            force_pull=(0.001, 0.004),
            ensure_inside_axes=True,
            expand_axes=True,
        )

        fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
        fig.savefig(plot_pdf, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] wrote {plot_path}")
    print(f"[OK] wrote {plot_pdf}")


if __name__ == "__main__":
    main()
