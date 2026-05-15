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
- forest_plot_rq3.png

Edit constants at the top to filter models, regimes, or source/target datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_analysis.active._statistical_helpers import (
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
    means = (
        df.groupby(["family", "source_dataset_regime"], as_index=False)
        .agg(
            mean_f1=("target_f1", "mean"),
            std_f1=("target_f1", "std"),
            n_obs=("target_f1", "count"),
        )
        .sort_values(["family", "source_dataset_regime"])
        .reset_index(drop=True)
    )
    return means


def plot_family_regime(means: pd.DataFrame, out_path: Path) -> None:
    if means.empty:
        return
    families = sorted(means["family"].astype(str).unique())
    regimes = ["solo", "mixed"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    width = 0.36
    x = np.arange(len(families))
    palette = {"solo": "#3a76af", "mixed": "#c44e52"}
    for i, regime in enumerate(regimes):
        vals = []
        for fam in families:
            row = means[(means["family"] == fam) & (means["source_dataset_regime"] == regime)]
            vals.append(float(row["mean_f1"].iloc[0]) if not row.empty else float("nan"))
        ax.bar(x + (i - 0.5) * width, vals, width, label=regime, color=palette.get(regime))
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("Mean OOD F1")
    ax.set_title("RQ3: Mean OOD F1 by family and regime")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="regime")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
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
    lines.append("- `mean_by_regime_family.csv`: cell means used for the bar plot.")
    lines.append("- `forest_plot_rq3.png`: bar plot of regime-by-family means.")
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

    plot_family_regime(means, out_dir / "forest_plot_rq3.png")

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
