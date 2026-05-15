"""Statistical analysis for RQ1.

RQ1: In-domain F1 systematically exceeds mean OOD F1 (per model).

- Null (H1_0): The per-checkpoint difference ``in_domain_f1 - mean_ood_f1``
  has a median of zero.
- Alternative (H1_1): The median difference is positive (in-domain F1 is
  greater than mean OOD F1).

Pipeline:
1. Load the master joined CSV produced by
   ``build_generalization_master_table.py``.
2. Apply the user-facing filters at the top of this file (models, regimes,
   source datasets used for training, target datasets evaluated).
3. Build per-checkpoint paired observations
   ``(in_domain_f1, mean_ood_f1)``.
4. Run a one-sided Wilcoxon signed-rank test per model, plus paired Cohen's
   d and a bootstrap 95% CI on the mean difference.
5. Apply Benjamini-Hochberg FDR correction across models.
6. Render a decision summary, per-model CSV, and a forest plot.

Outputs (workspace/statistical_results/RQ1):
- decision_summary.md
- per_checkpoint_pairs.csv
- per_model_results.csv
- by_regime.csv
- forest_plot_rq1.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
    bh_fdr,
    bootstrap_ci_mean,
    describe_filters,
    df_to_markdown,
)


# =============================================================================
# CONSTANTS - edit to filter what is included in the test
# =============================================================================

# None means "all models". Set to a tuple of strings to whitelist.
MODELS_INCLUDE: tuple[str, ...] | None = None

# Models to remove from the analysis even if present in MODELS_INCLUDE.
MODELS_EXCLUDE: tuple[str, ...] = ('EAT')

# Regimes contributing observations (any subset of {"solo", "mixed"}).
REGIMES_INCLUDE: tuple[str, ...] = ("solo", "mixed")

# Source datasets to keep (training data identity); None means all.
# Examples: ("herlev",), ("herlev_sipakmed", "riva_herlev"), etc.
SOURCE_DATASETS_INCLUDE: tuple[str, ...] | None = None

# Source datasets to drop (evaluated against the master CSV's
# ``source_dataset`` column).
SOURCE_DATASETS_EXCLUDE: tuple[str, ...] = ()

# Target datasets to keep when computing the mean OOD F1 per checkpoint.
# None means all OOD targets in the master CSV.
TARGET_DATASETS_INCLUDE: tuple[str, ...] | None = None
TARGET_DATASETS_EXCLUDE: tuple[str, ...] = ()

# Acceptance criteria for H1_1 (from research_questions_and_hypotheses.md).
ALPHA: float = 0.05
RQ1_MODEL_FRACTION: float = 0.75
RQ1_COHEN_D: float = 0.5
N_BOOTSTRAP: int = 10_000
BOOTSTRAP_SEED: int = 42

# =============================================================================

MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/statistical_results/RQ1")

CKPT_KEYS = [
    "cfg_id",
    "source_dataset",
    "source_dataset_regime",
    "source_model",
    "source_origin",
    "source_fold",
    "best_epoch",
]


@dataclass
class TestRow:
    n: int
    mean: float
    median: float
    ci_low: float
    ci_high: float
    cohens_d: float
    wilcoxon_stat: float
    pvalue: float


def load_data() -> pd.DataFrame:
    path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing master CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in ("target_f1", "in_domain_f1_mean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target_f1", "in_domain_f1_mean"]).copy()
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
    return df


def build_pairs(master: pd.DataFrame) -> pd.DataFrame:
    """Per-checkpoint paired (in-domain F1, mean OOD F1)."""
    in_domain = master.groupby(CKPT_KEYS, as_index=False)["in_domain_f1_mean"].mean()
    in_domain = in_domain.rename(columns={"in_domain_f1_mean": "in_domain_f1"})
    ood = master.copy()
    if "is_in_domain_target" in ood.columns:
        ood = ood[ood["is_in_domain_target"].astype(str).str.lower().isin({"false", "0"})]
    mean_ood = (
        ood.groupby(CKPT_KEYS, as_index=False)["target_f1"].mean()
        .rename(columns={"target_f1": "mean_ood_f1"})
    )
    pairs = in_domain.merge(mean_ood, on=CKPT_KEYS, how="inner")
    pairs = pairs.dropna(subset=["in_domain_f1", "mean_ood_f1"]).copy()
    pairs["diff_in_domain_minus_ood"] = pairs["in_domain_f1"] - pairs["mean_ood_f1"]
    return pairs


def cohens_d_paired(diffs: np.ndarray) -> float:
    arr = np.asarray(diffs, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan")
    sd = float(np.std(arr, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(arr) / sd)


def wilcoxon_one_sided_greater(diffs: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan"), float("nan")
    if np.allclose(arr, 0.0):
        return 0.0, 1.0
    try:
        result = stats.wilcoxon(arr, alternative="greater", zero_method="wilcox", correction=False)
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def summarize_diffs(diffs: np.ndarray) -> TestRow:
    arr = np.asarray(diffs, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else float("nan")
    median = float(np.median(arr)) if n else float("nan")
    ci_low, ci_high = bootstrap_ci_mean(arr, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED) if n else (float("nan"), float("nan"))
    cd = cohens_d_paired(arr)
    stat, pv = wilcoxon_one_sided_greater(arr)
    return TestRow(n=n, mean=mean, median=median, ci_low=ci_low, ci_high=ci_high,
                   cohens_d=cd, wilcoxon_stat=stat, pvalue=pv)


def per_model_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, sub in pairs.groupby("source_model", sort=True):
        diffs = sub["diff_in_domain_minus_ood"].to_numpy(dtype=float)
        s = summarize_diffs(diffs)
        rows.append(
            {
                "source_model": str(model),
                "n_checkpoints": s.n,
                "mean_diff": s.mean,
                "median_diff": s.median,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
                "cohens_d": s.cohens_d,
                "wilcoxon_stat": s.wilcoxon_stat,
                "p_value_one_sided": s.pvalue,
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_diff", ascending=False).reset_index(drop=True)
    summary["p_value_fdr"] = bh_fdr(summary["p_value_one_sided"].tolist())
    summary["meets_significance"] = summary["p_value_fdr"] < ALPHA
    summary["meets_effect_size"] = summary["cohens_d"] >= RQ1_COHEN_D
    summary["accept_per_model"] = summary["meets_significance"] & summary["meets_effect_size"]
    return summary


def plot_forest(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    df = summary.sort_values("mean_diff", ascending=True).reset_index(drop=True)
    fig_h = max(4.0, 0.3 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(7.4, fig_h), constrained_layout=True)
    y = np.arange(len(df))
    means = df["mean_diff"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    err = np.vstack([means - lo, hi - means])
    colors = ["#1b7a3e" if a else "#a8a8a8" for a in df["accept_per_model"]]
    ax.errorbar(means, y, xerr=err, fmt="o", ecolor="0.55", elinewidth=1.0,
                capsize=3, mfc="white", mec="black", lw=1.0, zorder=3)
    ax.scatter(means, y, c=colors, s=38, zorder=4, edgecolor="black", linewidths=0.5)
    ax.axvline(0.0, linestyle="--", color="0.55", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["source_model"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("Mean (in-domain F1 - mean OOD F1)")
    ax.set_title("RQ1 forest plot: per-model in-domain vs OOD difference")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, *, summary: pd.DataFrame, pairs: pd.DataFrame,
                  by_regime: pd.DataFrame, decision: bool) -> None:
    n_models = len(summary)
    n_accept = int(summary["accept_per_model"].sum())
    fraction = (n_accept / n_models) if n_models else 0.0
    overall = summarize_diffs(pairs["diff_in_domain_minus_ood"].to_numpy(dtype=float))

    lines: list[str] = []
    lines.append("# RQ1 - In-Domain Performance Overestimation\n")
    lines.append("## Hypothesis\n")
    lines.append("- Null (H1_0): For each model, the per-checkpoint difference "
                 "`in_domain_f1 - mean_ood_f1` has a median equal to zero.")
    lines.append("- Alternative (H1_1): The median difference is positive "
                 "(in-domain F1 is greater than mean OOD F1).\n")
    lines.append("## Acceptance Criteria\n")
    lines.append(f"- At least {RQ1_MODEL_FRACTION:.0%} of models with FDR-adjusted "
                 f"p-value < {ALPHA} and Cohen's d >= {RQ1_COHEN_D}.\n")
    lines.append("## Filter Configuration\n")
    lines.extend(describe_filters(
        models_include=MODELS_INCLUDE, models_exclude=MODELS_EXCLUDE,
        regimes_include=REGIMES_INCLUDE,
        source_datasets_include=SOURCE_DATASETS_INCLUDE,
        source_datasets_exclude=SOURCE_DATASETS_EXCLUDE,
        target_datasets_include=TARGET_DATASETS_INCLUDE,
        target_datasets_exclude=TARGET_DATASETS_EXCLUDE,
    ))
    lines.append("")
    lines.append("## Global Result\n")
    lines.append(f"- Models tested: {n_models}")
    lines.append(f"- Models meeting criteria: {n_accept} ({fraction:.0%})")
    lines.append(f"- Overall decision: {'ACCEPT H1_1' if decision else 'REJECT H1_1'}\n")
    lines.append("## Pooled Effect Across All Checkpoints\n")
    lines.append(f"- N checkpoints: {overall.n}")
    lines.append(f"- Mean diff: {overall.mean:.4f} (95% CI [{overall.ci_low:.4f}, {overall.ci_high:.4f}])")
    lines.append(f"- Median diff: {overall.median:.4f}")
    lines.append(f"- Cohen's d (paired): {overall.cohens_d:.3f}")
    lines.append(f"- Wilcoxon p-value (one-sided, greater): {overall.pvalue:.4g}\n")
    lines.append("## By Training Regime\n")
    lines.append(df_to_markdown(by_regime))
    lines.append("\n## Per-Model Decisions\n")
    cols = ["source_model", "n_checkpoints", "mean_diff", "ci_low", "ci_high",
            "cohens_d", "p_value_one_sided", "p_value_fdr", "accept_per_model"]
    lines.append(df_to_markdown(summary[cols]))
    lines.append("\n## Outputs\n")
    lines.append("- `per_checkpoint_pairs.csv`: paired (in-domain, OOD) per checkpoint.")
    lines.append("- `per_model_results.csv`: per-model statistics and decisions.")
    lines.append("- `by_regime.csv`: pooled differences split by training regime.")
    lines.append("- `forest_plot_rq1.png`: forest plot of per-model mean differences.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if df.empty:
        raise RuntimeError("No data after filters; relax MODELS_*/REGIMES_*/_DATASETS_* settings.")

    pairs = build_pairs(df)
    pairs.to_csv(out_dir / "per_checkpoint_pairs.csv", index=False)
    if pairs.empty:
        raise RuntimeError("No paired observations after building checkpoint pairs.")

    summary = per_model_summary(pairs)
    summary.to_csv(out_dir / "per_model_results.csv", index=False)

    by_regime = (
        pairs.groupby("source_dataset_regime", as_index=False)
        .agg(
            mean_diff=("diff_in_domain_minus_ood", "mean"),
            median_diff=("diff_in_domain_minus_ood", "median"),
            n_checkpoints=("diff_in_domain_minus_ood", "count"),
        )
    )
    by_regime.to_csv(out_dir / "by_regime.csv", index=False)

    plot_forest(summary, out_dir / "forest_plot_rq1.png")

    n_models = len(summary)
    fraction = (int(summary["accept_per_model"].sum()) / n_models) if n_models else 0.0
    decision = fraction >= RQ1_MODEL_FRACTION

    write_summary(out_dir / "decision_summary.md", summary=summary, pairs=pairs,
                  by_regime=by_regime, decision=decision)
    print(f"[OK] RQ1 outputs in {out_dir}")


if __name__ == "__main__":
    main()
