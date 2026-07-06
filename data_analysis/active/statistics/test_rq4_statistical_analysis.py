"""Statistical analysis for RQ4.

RQ4: Domain shift magnitude (FID) predicts cross-dataset F1.

- Null (H4_0): Spearman correlation between FID(source, target) and OOD F1 is
  zero.
- Alternative (H4_1): The correlation is negative (higher FID -> lower OOD F1).

Analyses produced:
1. Per-checkpoint FID-to-target computed from pairwise FID (default: minimum
   FID across source components for mixed regimes; configurable below).
2. Spearman correlation overall and stratified by training regime and
   architecture family, with bootstrap 95% CIs.
3. Two-panel distribution plot (solo vs mixed) of OOD F1 across FID levels.

Outputs (workspace/statistical_results/RQ4):
- decision_summary.md
- per_observation_fid_f1.csv
- correlation_results.csv
- mean_f1_by_regime_fid_level.csv
- distribution_fid_vs_f1.png

Edit constants at the top to filter models, regimes, source/target datasets,
or FID aggregation rules.
"""

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
from scipy import stats


from data_analysis.active.lib.statistical_helpers import (
    apply_filters,
    bootstrap_ci_mean,
    bootstrap_ci_paired_statistic,
    describe_filters,
    df_to_markdown,
    model_family,
)


# =============================================================================
# CONSTANTS - edit to filter models, regimes, dataset configs, or FID aggregation
# =============================================================================

MODELS_INCLUDE: tuple[str, ...] | None = None
MODELS_EXCLUDE: tuple[str, ...] = ('EAT')
REGIMES_INCLUDE: tuple[str, ...] = ("solo", "mixed")

SOURCE_DATASETS_INCLUDE: tuple[str, ...] | None = None
SOURCE_DATASETS_EXCLUDE: tuple[str, ...] = ()

TARGET_DATASETS_INCLUDE: tuple[str, ...] | None = None
TARGET_DATASETS_EXCLUDE: tuple[str, ...] = ()

# How to map mixed source -> target FID. Solo sources always use the direct
# pairwise FID. For mixed sources with components (a, b) and target c, we
# aggregate FID(a, c) and FID(b, c). Choose: "min", "mean", or "max".
FID_AGGREGATION_FOR_MIXED: str = "min"

# Significance level for the FDR-adjusted Spearman test.
ALPHA: float = 0.05
N_BOOTSTRAP: int = 10_000
BOOTSTRAP_SEED: int = 42

# =============================================================================

MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
PAIRWISE_FID_CSV = Path("workspace/analysis/domain_shift/pairwise_fid.csv")
OUT_DIR = Path("workspace/statistical_results/RQ4")

# Canonical pairwise FID values (low -> high) used to bin observations.
FID_CANONICAL: tuple[tuple[str, float], ...] = (
    ("low", 139.71573805959324),
    ("medium", 176.90400837001258),
    ("high", 209.64797687866402),
)
FID_LEVEL_ORDER: tuple[str, ...] = tuple(k for k, _ in FID_CANONICAL)
FID_LEVEL_LABELS: dict[str, str] = {"low": "Low", "medium": "Medium", "high": "High"}
REGIME_PANELS: tuple[tuple[str, str], ...] = (("solo", "(a) Solo"), ("mixed", "(b) Mixed"))


def load_pairwise_fid() -> dict[tuple[str, str], float]:
    path = (PAIRWISE_FID_CSV if PAIRWISE_FID_CSV.is_absolute() else _REPO_ROOT / PAIRWISE_FID_CSV).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing pairwise FID CSV: {path}")
    df = pd.read_csv(path)
    fid: dict[tuple[str, str], float] = {}
    for _, r in df.iterrows():
        a = str(r["dataset_a"])
        b = str(r["dataset_b"])
        v = float(r["fid"])
        fid[(a, b)] = v
        fid[(b, a)] = v
    return fid


def aggregate_fid(components: list[str], target: str, fid: dict[tuple[str, str], float]) -> float:
    values: list[float] = []
    for c in components:
        if c == target:
            continue
        v = fid.get((c, target))
        if v is None:
            continue
        values.append(float(v))
    if not values:
        return float("nan")
    if FID_AGGREGATION_FOR_MIXED == "min":
        return float(min(values))
    if FID_AGGREGATION_FOR_MIXED == "max":
        return float(max(values))
    if FID_AGGREGATION_FOR_MIXED == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"Unsupported FID_AGGREGATION_FOR_MIXED={FID_AGGREGATION_FOR_MIXED!r}")


def load_data() -> pd.DataFrame:
    path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing master CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["target_f1"] = pd.to_numeric(df["target_f1"], errors="coerce")
    df = df.dropna(subset=["target_f1"]).copy()
    df = apply_filters(
        df,
        models_include=MODELS_INCLUDE,
        models_exclude=MODELS_EXCLUDE,
        regimes_include=REGIMES_INCLUDE,
        source_datasets_include=SOURCE_DATASETS_INCLUDE,
        source_datasets_exclude=SOURCE_DATASETS_EXCLUDE,
        target_datasets_include=TARGET_DATASETS_INCLUDE,
        target_datasets_exclude=TARGET_DATASETS_EXCLUDE,
    )
    df["family"] = df["source_model"].astype(str).map(model_family)

    fid = load_pairwise_fid()
    fid_values: list[float] = []
    for _, r in df.iterrows():
        components = [c.strip() for c in str(r["source_dataset_components"]).split(",") if c.strip()]
        fid_values.append(aggregate_fid(components, str(r["target_dataset"]), fid))
    df["fid_to_target"] = fid_values
    df = df.dropna(subset=["fid_to_target"]).copy()
    df["fid_level"] = _assign_fid_level(df["fid_to_target"].to_numpy(dtype=float))
    return df


def _assign_fid_level(values: np.ndarray) -> list[str]:
    canonical = np.array([v for _, v in FID_CANONICAL], dtype=float)
    keys = [k for k, _ in FID_CANONICAL]
    idx = np.argmin(np.abs(values[:, None] - canonical[None, :]), axis=1)
    return [keys[i] for i in idx]


def mean_f1_by_regime_fid_level(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime, _panel in REGIME_PANELS:
        for level, fid_val in FID_CANONICAL:
            sub = df[
                (df["source_dataset_regime"].astype(str) == regime)
                & (df["fid_level"].astype(str) == level)
            ]
            if sub.empty:
                continue
            vals = sub["target_f1"].to_numpy(dtype=float)
            ci_lo, ci_hi = bootstrap_ci_mean(vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
            rows.append(
                {
                    "source_dataset_regime": regime,
                    "fid_level": level,
                    "fid_level_label": FID_LEVEL_LABELS[level],
                    "fid_canonical": fid_val,
                    "mean_f1": float(np.mean(vals)),
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "n_obs": int(vals.size),
                }
            )
    return pd.DataFrame(rows)


def spearman_with_ci(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size < 3 or y.size < 3:
        return {"rho": float("nan"), "p_two_sided": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": int(x.size)}
    res = stats.spearmanr(x, y)
    rho = float(res.correlation)
    p = float(res.pvalue)

    def _stat(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3:
            return float("nan")
        r = stats.spearmanr(a, b)
        return float(r.correlation)

    ci_lo, ci_hi = bootstrap_ci_paired_statistic(
        x, y, _stat, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    return {"rho": rho, "p_two_sided": p, "ci_low": ci_lo, "ci_high": ci_hi, "n": int(x.size)}


def stratified_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(label: str, sub: pd.DataFrame) -> None:
        x = sub["fid_to_target"].to_numpy(dtype=float)
        y = sub["target_f1"].to_numpy(dtype=float)
        s = spearman_with_ci(x, y)
        rows.append({"stratum": label, **s})

    add("overall", df)
    for regime, sub in df.groupby("source_dataset_regime", sort=True):
        add(f"regime={regime}", sub)
    for family, sub in df.groupby("family", sort=True):
        add(f"family={family}", sub)
    for (regime, family), sub in df.groupby(["source_dataset_regime", "family"], sort=True):
        add(f"regime={regime}|family={family}", sub)
    return pd.DataFrame(rows)


def plot_fid_distribution_panels(
    df: pd.DataFrame,
    corr_table: pd.DataFrame,
    out_path: Path,
) -> None:
    """Two-panel boxplot of OOD F1 by canonical FID level, stratified by training regime."""
    if df.empty:
        return

    fig_h = 2.35 * len(REGIME_PANELS) + 0.5
    fig, axes = plt.subplots(len(REGIME_PANELS), 1, figsize=(3.4, fig_h), sharey=True)
    if len(REGIME_PANELS) == 1:
        axes = [axes]

    x = np.arange(len(FID_LEVEL_ORDER))
    for ax, (regime, panel_label) in zip(axes, REGIME_PANELS, strict=True):
        sub = df[df["source_dataset_regime"].astype(str) == regime]
        box_data: list[np.ndarray] = []
        means: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []
        for level in FID_LEVEL_ORDER:
            vals = sub.loc[sub["fid_level"].astype(str) == level, "target_f1"].to_numpy(dtype=float)
            box_data.append(vals)
            if vals.size:
                m = float(np.mean(vals))
                lo, hi = bootstrap_ci_mean(vals, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
                means.append(m)
                yerr_lo.append(m - lo)
                yerr_hi.append(hi - m)
            else:
                means.append(float("nan"))
                yerr_lo.append(float("nan"))
                yerr_hi.append(float("nan"))

        bp = ax.boxplot(
            box_data,
            positions=x,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            boxprops={"facecolor": "0.92", "edgecolor": "black", "linewidth": 0.8},
            whiskerprops={"color": "0.45", "linewidth": 0.8},
            capprops={"color": "0.45", "linewidth": 0.8},
        )
        _ = bp  # silence linter
        valid_x = [i for i, m in enumerate(means) if not np.isnan(m)]
        if valid_x:
            vx = np.array(valid_x, dtype=float)
            vm = np.array([means[i] for i in valid_x], dtype=float)
            vlo = np.array([yerr_lo[i] for i in valid_x], dtype=float)
            vhi = np.array([yerr_hi[i] for i in valid_x], dtype=float)
            ax.errorbar(
                vx,
                vm,
                yerr=np.vstack([vlo, vhi]),
                fmt="D",
                color="black",
                ecolor="black",
                elinewidth=0.85,
                capsize=2.5,
                markersize=4.0,
                zorder=4,
            )

        rho_row = corr_table[corr_table["stratum"].astype(str) == f"regime={regime}"]
        if not rho_row.empty:
            rho = float(rho_row.iloc[0]["rho"])
            ax.text(
                0.98,
                0.97,
                f"$\\rho = {rho:.3f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([FID_LEVEL_LABELS[lvl] for lvl in FID_LEVEL_ORDER], fontsize=7)
        ax.set_title(panel_label, fontsize=8, loc="left", pad=4)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="y", labelsize=6.5)

    axes[0].set_ylabel("OOD F1", fontsize=7)
    fig.tight_layout(h_pad=0.65)
    top = axes[0].get_position()
    fig.suptitle(
        "Out-of-domain F1 by domain-shift (FID) level and training regime",
        fontsize=7.5,
        y=top.y1 + 0.045,
        va="bottom",
    )
    axes[-1].set_xlabel("Domain-shift level (FID)", fontsize=7)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_summary(path: Path, *, table: pd.DataFrame, decision: bool,
                  n_observations: int) -> None:
    overall_row = table[table["stratum"] == "overall"].iloc[0]
    lines: list[str] = []
    lines.append("# RQ4 - Domain Shift (FID) Predicts Cross-Dataset F1\n")
    lines.append("## Hypothesis\n")
    lines.append("- Null (H4_0): Spearman correlation between FID(source, target) "
                 "and OOD F1 is zero.")
    lines.append("- Alternative (H4_1): Correlation is negative (higher FID -> "
                 "lower OOD F1).\n")
    lines.append("## Acceptance Criteria\n")
    lines.append(f"- Overall Spearman rho is negative AND two-sided p-value < {ALPHA}, AND")
    lines.append("- Bootstrap 95% CI on rho excludes zero (upper bound < 0).\n")
    lines.append("## Configuration\n")
    lines.append(f"- FID aggregation for mixed sources: {FID_AGGREGATION_FOR_MIXED}")
    lines.extend(
        describe_filters(
            models_include=MODELS_INCLUDE,
            models_exclude=MODELS_EXCLUDE,
            regimes_include=REGIMES_INCLUDE,
            source_datasets_include=SOURCE_DATASETS_INCLUDE,
            source_datasets_exclude=SOURCE_DATASETS_EXCLUDE,
            target_datasets_include=TARGET_DATASETS_INCLUDE,
            target_datasets_exclude=TARGET_DATASETS_EXCLUDE,
        )
    )
    lines.append(f"- Total observations (after filtering): {n_observations}\n")
    lines.append("## Decision\n")
    lines.append(f"- Overall Spearman rho: {float(overall_row['rho']):.4f}")
    lines.append(f"- Overall p-value: {float(overall_row['p_two_sided']):.4g}")
    lines.append(f"- Bootstrap 95% CI: [{float(overall_row['ci_low']):.4f}, "
                 f"{float(overall_row['ci_high']):.4f}]")
    lines.append(f"- Decision: {'ACCEPT H4_1' if decision else 'REJECT H4_1'}\n")
    lines.append("## Stratified Correlations\n")
    lines.append(df_to_markdown(table))
    lines.append("\n## Outputs\n")
    lines.append("- `per_observation_fid_f1.csv`: per-row FID and OOD F1.")
    lines.append("- `correlation_results.csv`: stratified Spearman with CIs.")
    lines.append("- `mean_f1_by_regime_fid_level.csv`: cell means and 95% bootstrap CIs.")
    lines.append("- `distribution_fid_vs_f1.png`: two-panel FID-level distribution plot.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if df.empty:
        raise RuntimeError("No FID-mapped data after filtering.")
    cols_to_save = [
        "source_dataset",
        "source_dataset_regime",
        "source_dataset_components",
        "source_model",
        "family",
        "source_fold",
        "target_dataset",
        "fid_to_target",
        "fid_level",
        "target_f1",
    ]
    df[cols_to_save].to_csv(out_dir / "per_observation_fid_f1.csv", index=False)

    table = stratified_correlations(df)
    table.to_csv(out_dir / "correlation_results.csv", index=False)

    cell_means = mean_f1_by_regime_fid_level(df)
    cell_means.to_csv(out_dir / "mean_f1_by_regime_fid_level.csv", index=False)

    plot_fid_distribution_panels(df, table, out_dir / "distribution_fid_vs_f1.png")

    overall = table[table["stratum"] == "overall"].iloc[0]
    decision = (
        float(overall["rho"]) < 0
        and float(overall["p_two_sided"]) < ALPHA
        and float(overall["ci_high"]) < 0
    )

    write_summary(out_dir / "decision_summary.md", table=table, decision=decision,
                  n_observations=int(len(df)))
    print(f"[OK] RQ4 outputs in {out_dir}")


if __name__ == "__main__":
    main()
