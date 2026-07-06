"""Statistical analysis for RQ3.

RQ3: Architecture family interacts with training regime.

- Null (H3_0): Mixed-vs-solo OOD F1 difference does not depend on architecture
  family.
- Alternative (H3_1): The improvement from mixed-dataset training differs
  between CNN and transformer families.

Analyses produced:
1. OLS regression of OOD ``target_f1`` on ``regime`` (mixed indicator),
   ``family`` (transformer indicator), and their interaction, with
   cluster-robust standard errors clustered by source model.
2. A non-parametric permutation test on the interaction effect, permuting
   regime labels within each model.
3. Family-stratified mixed-vs-solo effect sizes (Cohen's d, Mann-Whitney U).

Outputs (workspace/statistical_results/RQ3):
- decision_summary.md
- ols_coefficients.csv
- family_stratified_effects.csv
- mean_by_regime_family.csv
- interaction_plot_rq3.png

Edit constants at the top to filter models, regimes, or source/target datasets.
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
    cluster_robust_ols,
    cohens_d_independent,
    describe_filters,
    df_to_markdown,
    model_family,
)


# =============================================================================
# CONSTANTS - edit to filter models, regimes, or dataset configurations
# =============================================================================

# None means "use all models present in the data". Otherwise a tuple of model
# names (matching ``source_model`` strings) to include.
MODELS_INCLUDE: tuple[str, ...] | None = None

# Models to exclude even if present in MODELS_INCLUDE.
MODELS_EXCLUDE: tuple[str, ...] = ('EAT')

# Regimes that contribute observations. RQ3 needs both regimes.
REGIMES_INCLUDE: tuple[str, ...] = ("solo", "mixed")

# Training source identity (``source_dataset`` in the master CSV).
SOURCE_DATASETS_INCLUDE: tuple[str, ...] | None = None
SOURCE_DATASETS_EXCLUDE: tuple[str, ...] = ()

# Evaluation targets (``target_dataset``).
TARGET_DATASETS_INCLUDE: tuple[str, ...] | None = None
TARGET_DATASETS_EXCLUDE: tuple[str, ...] = ()

# Number of permutations for the interaction test.
N_PERMUTATIONS: int = 5000
PERMUTATION_SEED: int = 42
N_BOOTSTRAP_MEAN: int = 10_000

# Plot order and display labels.
REGIME_ORDER: tuple[str, ...] = ("solo", "mixed")
REGIME_LABELS: dict[str, str] = {"solo": "Solo", "mixed": "Mixed"}
FAMILY_ORDER: tuple[str, ...] = ("cnn", "transformer")
FAMILY_LABELS: dict[str, str] = {"cnn": "CNN", "transformer": "Transformer-based"}

# Acceptance criteria for H3_1 (from research_questions_and_hypotheses.md).
ALPHA: float = 0.05
INTERACTION_ABS_EFFECT_THRESHOLD: float = 0.02

# =============================================================================

MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/statistical_results/RQ3")


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
    return df


def build_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regime_mixed = (df["source_dataset_regime"].astype(str) == "mixed").astype(float).to_numpy()
    family_transformer = (df["family"].astype(str) == "transformer").astype(float).to_numpy()
    interaction = regime_mixed * family_transformer
    intercept = np.ones_like(regime_mixed, dtype=float)
    X = np.column_stack([intercept, regime_mixed, family_transformer, interaction])
    y = df["target_f1"].to_numpy(dtype=float)
    clusters = df["source_model"].astype(str).to_numpy()
    return y, X, clusters


def fit_ols(df: pd.DataFrame) -> pd.DataFrame:
    y, X, clusters = build_design(df)
    beta, cov = cluster_robust_ols(y, X, clusters)
    se = np.sqrt(np.diag(cov))
    z = beta / np.where(se == 0, np.nan, se)
    p_two_sided = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    return pd.DataFrame(
        {
            "term": ["intercept", "regime_mixed", "family_transformer", "regime_mixed:family_transformer"],
            "estimate": beta,
            "std_error_cluster_model": se,
            "z_statistic": z,
            "p_two_sided": p_two_sided,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
    )


def permutation_interaction(df: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    """Permute regime labels within each model. Returns (observed, p, distribution)."""

    def interaction_effect(local: pd.DataFrame) -> float:
        means = (
            local.groupby(["source_dataset_regime", "family"], as_index=False)["target_f1"].mean()
            .pivot(index="source_dataset_regime", columns="family", values="target_f1")
        )
        # interaction = (mixed - solo on transformer) - (mixed - solo on cnn)
        try:
            tr = float(means.loc["mixed", "transformer"] - means.loc["solo", "transformer"])
            cn = float(means.loc["mixed", "cnn"] - means.loc["solo", "cnn"])
        except KeyError:
            return float("nan")
        return tr - cn

    observed = interaction_effect(df)
    rng = np.random.default_rng(PERMUTATION_SEED)
    dist = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        permuted = df.copy()
        permuted["source_dataset_regime"] = (
            permuted.groupby("source_model")["source_dataset_regime"]
            .transform(lambda s: rng.permutation(s.to_numpy()))
        )
        dist[i] = interaction_effect(permuted)
    valid = dist[~np.isnan(dist)]
    if valid.size == 0 or np.isnan(observed):
        return observed, float("nan"), dist
    p = float((np.sum(np.abs(valid) >= abs(observed)) + 1) / (valid.size + 1))
    return observed, p, dist


def family_stratified_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fam, sub in df.groupby("family", sort=True):
        mixed = sub.loc[sub["source_dataset_regime"] == "mixed", "target_f1"].to_numpy(dtype=float)
        solo = sub.loc[sub["source_dataset_regime"] == "solo", "target_f1"].to_numpy(dtype=float)
        if mixed.size == 0 or solo.size == 0:
            continue
        mean_diff = float(np.mean(mixed) - np.mean(solo))
        ci_lo, ci_hi = bootstrap_ci_mean(mixed - mixed.mean() + (mean_diff))  # placeholder, replaced below
        # Bootstrap CI for the mean difference (independent samples).
        rng = np.random.default_rng(PERMUTATION_SEED)
        n_boot = 10000
        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            m_idx = rng.integers(0, mixed.size, size=mixed.size)
            s_idx = rng.integers(0, solo.size, size=solo.size)
            boots[i] = float(np.mean(mixed[m_idx]) - np.mean(solo[s_idx]))
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))
        d = cohens_d_independent(mixed, solo)
        try:
            mw = stats.mannwhitneyu(mixed, solo, alternative="two-sided")
            p_mw = float(mw.pvalue)
        except ValueError:
            p_mw = float("nan")
        rows.append(
            {
                "family": fam,
                "n_mixed": int(mixed.size),
                "n_solo": int(solo.size),
                "mean_mixed": float(np.mean(mixed)),
                "mean_solo": float(np.mean(solo)),
                "mean_diff_mixed_minus_solo": mean_diff,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "cohens_d": d,
                "mann_whitney_p_two_sided": p_mw,
            }
        )
    return pd.DataFrame(rows)


def regime_family_means(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        for regime in REGIME_ORDER:
            sub = df[
                (df["family"].astype(str) == family)
                & (df["source_dataset_regime"].astype(str) == regime)
            ]
            if sub.empty:
                continue
            vals = sub["target_f1"].to_numpy(dtype=float)
            ci_lo, ci_hi = bootstrap_ci_mean(
                vals, n_boot=N_BOOTSTRAP_MEAN, seed=PERMUTATION_SEED
            )
            rows.append(
                {
                    "family": family,
                    "family_label": FAMILY_LABELS.get(family, family),
                    "source_dataset_regime": regime,
                    "regime_label": REGIME_LABELS.get(regime, regime),
                    "mean_f1": float(np.mean(vals)),
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "std_f1": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
                    "n_obs": int(vals.size),
                }
            )
    return pd.DataFrame(rows)


def plot_regime_family_interaction(means: pd.DataFrame, out_path: Path) -> None:
    """Interaction plot: regime on x-axis, mean OOD F1 on y, one line per family."""
    if means.empty:
        return

    fig_w, fig_h = 3.4, 2.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(len(REGIME_ORDER))
    family_styles = {
        "cnn": {"color": "black", "marker": "o", "linestyle": "-"},
        "transformer": {"color": "0.35", "marker": "s", "linestyle": "--"},
    }

    for family in FAMILY_ORDER:
        sub = means[means["family"].astype(str) == family].set_index("source_dataset_regime")
        if sub.empty:
            continue
        y = np.array([float(sub.loc[r, "mean_f1"]) for r in REGIME_ORDER])
        lo = np.array([float(sub.loc[r, "ci_low"]) for r in REGIME_ORDER])
        hi = np.array([float(sub.loc[r, "ci_high"]) for r in REGIME_ORDER])
        yerr = np.vstack([y - lo, hi - y])
        style = family_styles.get(family, {"color": "black", "marker": "o", "linestyle": "-"})
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{style['marker']}{style['linestyle']}",
            color=style["color"],
            ecolor=style["color"],
            elinewidth=0.85,
            capsize=3.0,
            markersize=5.0,
            mfc=style["color"],
            mec=style["color"],
            lw=1.0,
            label=FAMILY_LABELS.get(family, family),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=7)
    ax.set_ylabel("Mean OOD F1", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Architecture family", fontsize=6.5, title_fontsize=6.5, loc="best")
    fig.tight_layout()
    top = ax.get_position()
    fig.suptitle(
        "Mean out-of-domain F1 by architecture family and training regime",
        fontsize=7.5,
        y=top.y1 + 0.06,
        va="bottom",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_summary(path: Path, *, ols_table: pd.DataFrame, observed: float,
                  perm_p: float, fam_effects: pd.DataFrame, means: pd.DataFrame,
                  decision: bool) -> None:
    interaction_row = ols_table[ols_table["term"] == "regime_mixed:family_transformer"].iloc[0]
    lines: list[str] = []
    lines.append("# RQ3 - Architecture Family x Training Regime Interaction\n")
    lines.append("## Hypothesis\n")
    lines.append("- Null (H3_0): The expected difference `mixed_ood_f1 - solo_ood_f1` "
                 "is the same across architecture families.")
    lines.append("- Alternative (H3_1): The expected difference depends on architecture family.\n")
    lines.append("## Acceptance Criteria\n")
    lines.append(f"- Cluster-robust Wald p-value on the interaction term < {ALPHA}, AND")
    lines.append(f"- Absolute interaction coefficient >= {INTERACTION_ABS_EFFECT_THRESHOLD}.\n")
    lines.append("## Filter Configuration\n")
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
    lines.append("")
    lines.append("## Decision\n")
    lines.append(f"- Interaction estimate: {float(interaction_row['estimate']):.4f}")
    lines.append(f"- Interaction CR1 SE: {float(interaction_row['std_error_cluster_model']):.4f}")
    lines.append(f"- Wald two-sided p-value: {float(interaction_row['p_two_sided']):.4g}")
    lines.append(f"- Permutation two-sided p-value: {perm_p:.4g}")
    lines.append(f"- Permutation observed effect: {observed:.4f}")
    lines.append(f"- Decision: {'ACCEPT H3_1' if decision else 'REJECT H3_1'}\n")
    lines.append("## OLS Coefficients (cluster-robust by source_model)\n")
    lines.append(df_to_markdown(ols_table))
    lines.append("\n## Family-Stratified Mixed vs Solo Effects\n")
    lines.append(df_to_markdown(fam_effects))
    lines.append("\n## Mean OOD F1 by family and regime\n")
    lines.append(df_to_markdown(means))
    lines.append("\n## Outputs\n")
    lines.append("- `ols_coefficients.csv`: OLS estimates with cluster-robust SE.")
    lines.append("- `family_stratified_effects.csv`: Cohen's d and Mann-Whitney per family.")
    lines.append("- `mean_by_regime_family.csv`: cell means and 95% bootstrap CIs.")
    lines.append("- `interaction_plot_rq3.png`: regime x family interaction plot.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if df.empty:
        raise RuntimeError(
            "No data after applying filters; check MODELS_*, REGIMES_INCLUDE, "
            "SOURCE_DATASETS_*, TARGET_DATASETS_*."
        )

    ols_table = fit_ols(df)
    ols_table.to_csv(out_dir / "ols_coefficients.csv", index=False)

    observed, perm_p, _dist = permutation_interaction(df)

    fam_effects = family_stratified_effects(df)
    fam_effects.to_csv(out_dir / "family_stratified_effects.csv", index=False)

    means = regime_family_means(df)
    means.to_csv(out_dir / "mean_by_regime_family.csv", index=False)

    plot_regime_family_interaction(means, out_dir / "interaction_plot_rq3.png")

    interaction_row = ols_table[ols_table["term"] == "regime_mixed:family_transformer"].iloc[0]
    p_value = float(interaction_row["p_two_sided"])
    abs_effect = float(abs(interaction_row["estimate"]))
    decision = (p_value < ALPHA) and (abs_effect >= INTERACTION_ABS_EFFECT_THRESHOLD)

    write_summary(
        out_dir / "decision_summary.md",
        ols_table=ols_table,
        observed=observed,
        perm_p=perm_p,
        fam_effects=fam_effects,
        means=means,
        decision=decision,
    )
    print(f"[OK] RQ3 outputs in {out_dir}")


if __name__ == "__main__":
    main()
