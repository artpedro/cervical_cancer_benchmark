from __future__ import annotations

"""
Two-panel efficiency vs OOD generalization plot (solo vs mixed training).

Per model and training regime:
  - y: mean out-of-domain abnormal-class F1 (all held-out targets and folds)
  - x: mean latency in ms/image (latency_mean_ms_last_k / latency_batch_size)
  - marker size: mean parameter count
  - marker color: CNN vs transformer-based architecture family

Selected models are labeled per panel; combined legend (family + parameters) in the lower-right.
"""

import sys
from pathlib import Path

_REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from adjustText import adjust_text
from data_analysis.active.lib.statistical_helpers import model_family

MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
EFF_CSV = Path("workspace/analysis/efficiency_profile/per_checkpoint_efficiency.csv")
OUT_DIR = Path("workspace/analysis/efficiency_profile/plots")
OUT_CSV = Path("workspace/analysis/efficiency_profile/ood_f1_vs_latency_per_model.csv")

REGIME_PANELS: tuple[tuple[str, str], ...] = (
    ("solo", "(a) Solo-source training"),
    ("mixed", "(b) Mixed-source training"),
)

# source_model names to label in each panel (manuscript-selected)
REGIME_LABEL_MODELS: dict[str, tuple[str, ...]] = {
    "solo": (
        "MobileViT v2 100",
        "FastViT T8",
        "LeViT 128s",
        "MobileNet v4 Conv Small",
        "EfficientFormerV2 S2",
    ),
    "mixed": (
        "MobileNet v4 Conv Small",
        "LeViT 128s",
        "MobileViT v2 100",
        "EfficientFormerV2 S1",
        "FastViT T8",
        "SqueezeNet 1.1",
    ),
}

# On-plot label text (defaults to MODEL_SHORT_NAMES when absent)
LABEL_DISPLAY: dict[str, str] = {
    "MobileViT v2 100": "MobileViTv2",
    "LeViT 128s": "LeViT-128s",
    "MobileNet v4 Conv Small": "MobileNetV4",
    "EfficientFormerV2 S2": "EfficientFormerV2-S2",
    "EfficientFormerV2 S1": "EfficientFormerV2-S1",
    "FastViT T8": "FastViT-T8",
}
COLOR_CNN = "#3a76af"
COLOR_TRANSFORMER = "#c44e52"
FAMILY_COLORS = {"cnn": COLOR_CNN, "transformer": COLOR_TRANSFORMER}
FAMILY_LABELS = {"cnn": "CNN", "transformer": "Transformer-based"}

MODEL_SHORT_NAMES: dict[str, str] = {
    "EfficientFormerV2 S0": "EffF-S0",
    "EfficientFormerV2 S1": "EffF-S1",
    "EfficientFormerV2 S2": "EffF-S2",
    "EfficientNet-B0": "EN-B0",
    "EfficientNet-B1": "EN-B1",
    "EfficientNet-B2": "EN-B2",
    "EfficientNet-B3": "EN-B3",
    "FastViT T8": "FastViT-T8",
    "LeViT 128s": "LeViT 128s",
    "MobileNet v2": "MBNet v2",
    "MobileNet v4 Conv Small": "MBNet v4",
    "MobileViT v2 100": "MViTv2",
    "ShuffleNet v2 1.0": "ShuffleNet",
    "SqueezeNet 1.1": "SqueezeNet",
    "iFormer-M": "iFormer-M",
    "EAT": "EAT",
}

FIG_W = 3.54
PANEL_H = 2.65
DPI = 300
DEFAULT_LATENCY_BATCH_SIZE = 16
MARKER_S_LO = 36.0
MARKER_S_HI = 280.0

LABEL_CONNECTOR_COLOR = "0.55"
LABEL_CONNECTOR_LW = 0.55
LABEL_CONNECTOR_PAD_PT = 1.5

# Sole exception: not placed above the bubble.
LABEL_PLACEMENT_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("mixed", "EfficientFormerV2 S1"): "right",
}

# Extra (dx, dy) in data coords applied after adjust_text.
LABEL_FINAL_NUDGE: dict[tuple[str, str], tuple[float, float]] = {
    ("mixed", "FastViT T8"): (0.0, 0.007),
}


def _latency_batch_size(eff: pd.DataFrame) -> int:
    if "latency_batch_size" not in eff.columns:
        return DEFAULT_LATENCY_BATCH_SIZE
    vals = pd.to_numeric(eff["latency_batch_size"], errors="coerce").dropna()
    if vals.empty:
        return DEFAULT_LATENCY_BATCH_SIZE
    mode = vals.mode()
    return int(mode.iloc[0]) if len(mode) else int(vals.iloc[0])


def _build_per_model_regime_table(master: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    work = master.copy()
    work["target_f1"] = pd.to_numeric(work["target_f1"], errors="coerce")
    work["latency_mean_ms"] = pd.to_numeric(work["latency_mean_ms"], errors="coerce")
    work["params_count_mean"] = pd.to_numeric(work["params_count_mean"], errors="coerce")
    work = work.dropna(subset=["target_f1", "latency_mean_ms", "params_count_mean"])

    ckpt = (
        work.groupby(
            ["source_model", "source_dataset_regime", "cfg_id"],
            as_index=False,
        )
        .agg(
            mean_ood_f1=("target_f1", "mean"),
            latency_mean_ms=("latency_mean_ms", "first"),
            params_count_mean=("params_count_mean", "first"),
        )
    )
    out = (
        ckpt.groupby(["source_model", "source_dataset_regime"], as_index=False)
        .agg(
            mean_ood_f1=("mean_ood_f1", "mean"),
            latency_mean_ms=("latency_mean_ms", "mean"),
            params_count_mean=("params_count_mean", "mean"),
            n_checkpoints=("cfg_id", "nunique"),
        )
    )
    out["latency_ms_per_image"] = out["latency_mean_ms"] / float(batch_size)
    out["latency_batch_size"] = batch_size
    out["family"] = out["source_model"].astype(str).map(model_family)
    return out


def _marker_sizes(
    params: np.ndarray,
    *,
    p_lo: float,
    p_hi: float,
    s_lo: float = MARKER_S_LO,
    s_hi: float = MARKER_S_HI,
) -> np.ndarray:
    span = p_hi - p_lo
    if span <= 0:
        return np.full(len(params), (s_lo + s_hi) / 2.0)
    t = (params - p_lo) / span
    return s_lo + t * (s_hi - s_lo)


def _param_count_refs(p_lo: float, p_hi: float) -> list[float]:
    span = p_hi - p_lo
    if span <= 0:
        return [p_lo, p_lo, p_lo]
    return [p_lo, (p_lo + p_hi) / 2.0, p_hi]


def _params_legend_label(n: float) -> str:
    return f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n / 1e3:.0f} k"


def _scatter_proxy(
    ax: plt.Axes,
    *,
    s: float,
    color: str,
    label: str,
) -> plt.PathCollection:
    return ax.scatter(
        [],
        [],
        s=s,
        c=color,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.4,
        label=label,
    )


def _panel_legend(
    ax: plt.Axes,
    *,
    p_lo: float,
    p_hi: float,
) -> plt.Legend:
    """Single legend on panel (b): family colors + sizes via scatter proxies matching data bubbles."""
    mid_s = (MARKER_S_LO + MARKER_S_HI) / 2.0
    family_handles = [
        _scatter_proxy(ax, s=mid_s, color=FAMILY_COLORS[fam], label=FAMILY_LABELS[fam])
        for fam in ("cnn", "transformer")
    ]
    refs = _param_count_refs(p_lo, p_hi)
    ref_s = _marker_sizes(np.asarray(refs, dtype=float), p_lo=p_lo, p_hi=p_hi)
    size_handles = [
        _scatter_proxy(ax, s=float(sv), color="0.50", label=_params_legend_label(rv))
        for sv, rv in zip(ref_s, refs)
    ]
    leg = ax.legend(
        handles=family_handles + size_handles,
        loc="lower right",
        frameon=True,
        edgecolor="0.78",
        facecolor="white",
        framealpha=0.92,
        fontsize=5.5,
        borderpad=0.4,
        labelspacing=0.55,
        handletextpad=0.5,
        ncol=1,
        markerscale=1.0,
    )
    leg.get_frame().set_linewidth(0.45)
    leg.set_zorder(10)
    return leg


def _display_label(source_model: str) -> str:
    name = str(source_model)
    return LABEL_DISPLAY.get(name, MODEL_SHORT_NAMES.get(name, name))


def _ray_bbox_intersection(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    bbox,
) -> tuple[float, float]:
    """First intersection of ray (x0, y0) -> (x1, y1) with a data-space bbox (t > 0)."""
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        cx = min(max(x0, bbox.x0), bbox.x1)
        cy = min(max(y0, bbox.y0), bbox.y1)
        return cx, cy

    hits: list[tuple[float, float, float]] = []
    if abs(dx) >= 1e-12:
        for xe in (bbox.x0, bbox.x1):
            t = (xe - x0) / dx
            if t > 1e-9:
                ye = y0 + t * dy
                if bbox.y0 - 1e-9 <= ye <= bbox.y1 + 1e-9:
                    hits.append((t, xe, ye))
    if abs(dy) >= 1e-12:
        for ye in (bbox.y0, bbox.y1):
            t = (ye - y0) / dy
            if t > 1e-9:
                xe = x0 + t * dx
                if bbox.x0 - 1e-9 <= xe <= bbox.x1 + 1e-9:
                    hits.append((t, xe, ye))
    if not hits:
        cx = min(max(x0, bbox.x0), bbox.x1)
        cy = min(max(y0, bbox.y0), bbox.y1)
        return cx, cy
    _, xe, ye = min(hits, key=lambda item: item[0])
    return xe, ye


def _connector_end_on_text(
    ax: plt.Axes,
    x0: float,
    y0: float,
    text,
    *,
    pad_pts: float = LABEL_CONNECTOR_PAD_PT,
) -> tuple[float, float]:
    renderer = ax.figure.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
    tcx = 0.5 * (bbox.x0 + bbox.x1)
    tcy = 0.5 * (bbox.y0 + bbox.y1)
    xe, ye = _ray_bbox_intersection(x0, y0, tcx, tcy, bbox)
    end_disp = ax.transData.transform((xe, ye))
    anchor_disp = ax.transData.transform((x0, y0))
    vec = anchor_disp - end_disp
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        end_disp = end_disp + vec / norm * pad_pts
    return tuple(ax.transData.inverted().transform(end_disp))


def _draw_label_connectors(
    ax: plt.Axes,
    texts: list,
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
) -> None:
    for text, x0, y0 in zip(texts, anchor_x, anchor_y, strict=True):
        xe, ye = _connector_end_on_text(ax, float(x0), float(y0), text)
        ax.plot(
            [x0, xe],
            [y0, ye],
            color=LABEL_CONNECTOR_COLOR,
            lw=LABEL_CONNECTOR_LW,
            solid_capstyle="round",
            zorder=4,
            clip_on=False,
        )


def _initial_label_position(
    model: str,
    regime: str,
    x: float,
    y: float,
    *,
    x_span: float,
    y_span: float,
) -> tuple[float, float, str, str]:
    placement = LABEL_PLACEMENT_EXCEPTIONS.get((regime, model), "above")
    if placement == "right":
        return x + x_span * 0.07, y, "left", "center"
    return x, y + y_span * 0.048, "center", "bottom"


def _apply_label_nudges(
    texts: list,
    models: list[str],
    regime: str,
) -> None:
    for text, model in zip(texts, models, strict=True):
        nudge = LABEL_FINAL_NUDGE.get((regime, model))
        if not nudge:
            continue
        x, y = text.get_position()
        text.set_position((x + nudge[0], y + nudge[1]))


def _draw_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    panel_title: str,
    p_lo: float,
    p_hi: float,
) -> tuple[float, float]:
    """Scatter + axes styling; returns (x_span, y_span) for label placement."""
    if sub.empty:
        return 0.0, 0.0

    x = sub["latency_ms_per_image"].to_numpy(dtype=float)
    y = sub["mean_ood_f1"].to_numpy(dtype=float)
    p = sub["params_count_mean"].to_numpy(dtype=float)
    s = _marker_sizes(p, p_lo=p_lo, p_hi=p_hi)

    for fam in ("cnn", "transformer"):
        m = sub["family"].astype(str) == fam
        if not m.any():
            continue
        idx = np.flatnonzero(m.to_numpy())
        ax.scatter(
            x[idx],
            y[idx],
            s=s[idx],
            alpha=0.82,
            c=FAMILY_COLORS[fam],
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = max(x_span * 0.12, 0.25)
    y_pad = max(y_span * 0.10, 0.01)
    ax.set_xlim(x_min - x_pad, x_max + x_pad * 1.35)
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 1.22)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax.set_title(panel_title, fontsize=7.5, loc="left", pad=3)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.35, color="#c8c8c8", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return x_span, y_span


def _place_panel_labels(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    regime: str,
    label_models: tuple[str, ...],
    x_span: float,
    y_span: float,
) -> None:
    if sub.empty or x_span <= 0:
        return

    x = sub["latency_ms_per_image"].to_numpy(dtype=float)
    y = sub["mean_ood_f1"].to_numpy(dtype=float)
    label_set = set(label_models)
    model_names = sub["source_model"].astype(str).tolist()
    label_idx = [i for i, name in enumerate(model_names) if name in label_set]
    if not label_idx:
        return

    anchor_x = x[label_idx]
    anchor_y = y[label_idx]
    texts = []
    labeled_models: list[str] = []
    for i in label_idx:
        model = model_names[i]
        labeled_models.append(model)
        tx, ty, ha, va = _initial_label_position(
            model,
            regime,
            float(x[i]),
            float(y[i]),
            x_span=x_span,
            y_span=y_span,
        )
        texts.append(
            ax.text(
                tx,
                ty,
                _display_label(model),
                fontsize=5.5,
                fontweight="medium",
                color="0.12",
                ha=ha,
                va=va,
                zorder=5,
            )
        )
    ax.figure.canvas.draw()
    adjust_text(
        texts,
        x=anchor_x,
        y=anchor_y,
        ax=ax,
        expand=(1.4, 1.6),
        force_text=(0.35, 0.55),
        force_static=(40, 25),
        force_pull=(0.02, 0.08),
        only_move={"text": "xy", "points": "xy", "objects": "xy"},
        ensure_inside_axes=True,
        expand_axes=True,
    )
    _apply_label_nudges(texts, labeled_models, regime)
    ax.figure.canvas.draw()
    _draw_label_connectors(ax, texts, anchor_x, anchor_y)


def main() -> None:
    master_path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    eff_path = (EFF_CSV if EFF_CSV.is_absolute() else _REPO_ROOT / EFF_CSV).resolve()
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master CSV: {master_path}")

    master = pd.read_csv(master_path, low_memory=False)
    eff = pd.read_csv(eff_path, low_memory=False) if eff_path.exists() else pd.DataFrame()
    batch_size = _latency_batch_size(eff)

    df = _build_per_model_regime_table(master, batch_size)
    if df.empty:
        raise RuntimeError("No models after building OOD F1 vs latency table.")

    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = (OUT_CSV if OUT_CSV.is_absolute() else _REPO_ROOT / OUT_CSV).resolve()
    df.sort_values(["source_dataset_regime", "mean_ood_f1"], ascending=[True, False]).to_csv(
        out_csv, index=False
    )
    print(f"[OK] wrote {out_csv}")

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
    }

    fig_h = PANEL_H * len(REGIME_PANELS) + 0.15
    plot_png = out_dir / "ood_f1_vs_latency.png"
    plot_pdf = out_dir / "ood_f1_vs_latency.pdf"

    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(REGIME_PANELS), 1, figsize=(FIG_W, fig_h))
        if len(REGIME_PANELS) == 1:
            axes = [axes]

        p_all = df["params_count_mean"].to_numpy(dtype=float)
        p_lo = float(np.min(p_all))
        p_hi = float(np.max(p_all))

        panel_spans: list[tuple[float, float]] = []
        for ax, (regime, panel_title) in zip(axes, REGIME_PANELS, strict=True):
            sub = df[df["source_dataset_regime"].astype(str) == regime].copy()
            panel_spans.append(
                _draw_panel(ax, sub, panel_title=panel_title, p_lo=p_lo, p_hi=p_hi)
            )

        axes[-1].set_xlabel("Latency (ms / image)", fontsize=7.5)
        axes[0].set_ylabel("Mean OOD F1", fontsize=7.5)

        for ax, (regime, _), (x_span, y_span) in zip(axes, REGIME_PANELS, panel_spans, strict=True):
            sub = df[df["source_dataset_regime"].astype(str) == regime].copy()
            _place_panel_labels(
                ax,
                sub,
                regime=regime,
                label_models=REGIME_LABEL_MODELS[regime],
                x_span=x_span,
                y_span=y_span,
            )

        fig.tight_layout(h_pad=0.85)
        _panel_legend(axes[-1], p_lo=p_lo, p_hi=p_hi)
        fig.savefig(plot_png, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(plot_pdf, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    print(f"[OK] wrote {plot_png}")
    print(f"[OK] wrote {plot_pdf}")


if __name__ == "__main__":
    main()
