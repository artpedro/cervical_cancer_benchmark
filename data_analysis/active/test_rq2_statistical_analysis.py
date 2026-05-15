"""Statistical analysis for RQ2.

RQ2: For each model, mixed-dataset training improves OOD F1 on the held-out
target compared to the best matched solo regime (per model x target).

- Null (H2_0): For each model and held-out target c, the difference
  ``mixed_ood_f1(c) - best_solo_ood_f1(c)`` has a median of zero, where
  ``best_solo_ood_f1`` is the maximum target F1 across solo runs trained on
  one of the components of the mixed pair.
- Alternative (H2_1): The median difference is positive in favor of mixed
  training.

Pipeline:
1. Load the master joined CSV.
2. Apply user-facing filters (models, mixed sources, target datasets).
   Both the ``mixed`` and ``solo`` rows must be available because the test
   compares them; therefore there is no regime filter.
3. For each mixed checkpoint with target ``c`` not in its source components,
   pair it with solo runs of the same model and fold trained on each
   component a, b in turn, evaluated on the same target c, taking the better
   of the two solos.
4. Per (model, target c) compute Wilcoxon signed-rank test on the paired
   differences across folds, plus Cohen's d and a bootstrap 95% CI.
5. Apply Benjamini-Hochberg FDR correction across (model, target) cells.
6. Decide per model: accept H2_1 if at least N targets pass criteria.

Outputs (workspace/statistical_results/RQ2):
- decision_summary.md
- per_pair_observations.csv
- per_model_target_results.csv
- per_model_decisions.csv
- by_target.csv
- forest_plot_rq2.png
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

# Models to exclude from the analysis.
MODELS_EXCLUDE: tuple[str, ...] = ('EAT')

# Mixed source datasets to include (e.g. ("herlev_sipakmed", "riva_herlev")).
# None means use every mixed source present in the master CSV.
MIXED_SOURCES_INCLUDE: tuple[str, ...] | None = None
MIXED_SOURCES_EXCLUDE: tuple[str, ...] = ()

# Solo source datasets considered when looking up matched solo baselines.
# None means use every solo source present in the master CSV.
SOLO_SOURCES_INCLUDE: tuple[str, ...] | None = None
SOLO_SOURCES_EXCLUDE: tuple[str, ...] = ()

# Held-out target datasets for which the test is computed. None means all.
TARGET_DATASETS_INCLUDE: tuple[str, ...] | None = None
TARGET_DATASETS_EXCLUDE: tuple[str, ...] = ()

# Acceptance criteria for H2_1.
ALPHA: float = 0.05
RQ2_TARGETS_REQUIRED: int = 2
RQ2_COHEN_D: float = 0.3
N_BOOTSTRAP: int = 10_000
BOOTSTRAP_SEED: int = 42

# =============================================================================

MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/statistical_results/RQ2")


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
    df["target_f1"] = pd.to_numeric(df["target_f1"], errors="coerce")
    df = df.dropna(subset=["target_f1"]).copy()

    # Apply model and target filters first; the regime filter is built-in: we
    # need both mixed and solo rows for the paired contrast.
    df = apply_filters(
        df,
        models_include=MODELS_INCLUDE,
        models_exclude=MODELS_EXCLUDE,
        regimes_include=("solo", "mixed"),
        target_datasets_include=TARGET_DATASETS_INCLUDE,
        target_datasets_exclude=TARGET_DATASETS_EXCLUDE,
    )
    return df


def build_pairs(master: pd.DataFrame) -> pd.DataFrame:
    mixed = master[master["source_dataset_regime"].astype(str) == "mixed"].copy()
    solo = master[master["source_dataset_regime"].astype(str) == "solo"].copy()
    if MIXED_SOURCES_INCLUDE:
        mixed = mixed[mixed["source_dataset"].astype(str).isin(set(MIXED_SOURCES_INCLUDE))]
    if MIXED_SOURCES_EXCLUDE:
        mixed = mixed[~mixed["source_dataset"].astype(str).isin(set(MIXED_SOURCES_EXCLUDE))]
    if SOLO_SOURCES_INCLUDE:
        solo = solo[solo["source_dataset"].astype(str).isin(set(SOLO_SOURCES_INCLUDE))]
    if SOLO_SOURCES_EXCLUDE:
        solo = solo[~solo["source_dataset"].astype(str).isin(set(SOLO_SOURCES_EXCLUDE))]

    rows: list[dict[str, object]] = []
    for _, mr in mixed.iterrows():
        components = [c.strip() for c in str(mr["source_dataset_components"]).split(",") if c.strip()]
        target = str(mr["target_dataset"])
        if target in components:
            continue
        match = solo[
            (solo["source_model"] == mr["source_model"])
            & (solo["source_origin"] == mr["source_origin"])
            & (solo["source_fold"] == mr["source_fold"])
            & (solo["target_dataset"].astype(str) == target)
            & (solo["source_dataset"].astype(str).isin(components))
        ]
        if match.empty:
            continue
        best_solo_f1 = float(pd.to_numeric(match["target_f1"], errors="coerce").max())
        sources_used = ",".join(sorted(set(match["source_dataset"].astype(str))))
        rows.append(
            {
                "source_model": str(mr["source_model"]),
                "source_origin": str(mr["source_origin"]),
                "fold": int(mr["source_fold"]),
                "target_dataset": target,
                "mixed_source": str(mr["source_dataset"]),
                "best_solo_sources": sources_used,
                "mixed_f1": float(mr["target_f1"]),
                "best_solo_f1": best_solo_f1,
                "diff_mixed_minus_solo": float(mr["target_f1"]) - best_solo_f1,
            }
        )
    return pd.DataFrame(rows)


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


def per_cell_results(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, target), sub in pairs.groupby(["source_model", "target_dataset"], sort=True):
        diffs = sub["diff_mixed_minus_solo"].to_numpy(dtype=float)
        s = summarize_diffs(diffs)
        rows.append(
            {
                "source_model": str(model),
                "target_dataset": str(target),
                "n_folds": s.n,
                "mean_diff_mixed_minus_solo": s.mean,
                "median_diff": s.median,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
                "cohens_d": s.cohens_d,
                "wilcoxon_stat": s.wilcoxon_stat,
                "p_value_one_sided": s.pvalue,
            }
        )
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells
    cells["p_value_fdr"] = bh_fdr(cells["p_value_one_sided"].tolist())
    cells["meets_significance"] = cells["p_value_fdr"] < ALPHA
    cells["meets_effect_size"] = cells["cohens_d"] >= RQ2_COHEN_D
    cells["cell_supports_h2"] = cells["meets_significance"] & cells["meets_effect_size"]
    return cells.sort_values(["source_model", "target_dataset"]).reset_index(drop=True)


def per_model_decision(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return cells
    decisions = (
        cells.groupby("source_model", as_index=False)
        .agg(
            n_targets=("target_dataset", "count"),
            n_targets_supporting=("cell_supports_h2", "sum"),
            mean_diff_overall=("mean_diff_mixed_minus_solo", "mean"),
            best_target_d=("cohens_d", "max"),
        )
    )
    decisions["accept_per_model"] = decisions["n_targets_supporting"] >= RQ2_TARGETS_REQUIRED
    return decisions.sort_values("mean_diff_overall", ascending=False).reset_index(drop=True)


def plot_forest(cells: pd.DataFrame, out_path: Path) -> None:
    if cells.empty:
        return
    targets = sorted(cells["target_dataset"].astype(str).unique())
    palette = {"herlev": "#3a76af", "sipakmed": "#c44e52", "riva": "#65a368"}
    df = cells.copy()
    order = (
        df.groupby("source_model")["mean_diff_mixed_minus_solo"].mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    df["source_model"] = pd.Categorical(df["source_model"].astype(str), categories=order, ordered=True)
    fig_h = max(4.5, 0.32 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(7.6, fig_h), constrained_layout=True)
    offsets = {t: i - (len(targets) - 1) / 2.0 for i, t in enumerate(targets)}
    spacing = 0.22
    for tgt in targets:
        sub = df[df["target_dataset"] == tgt]
        if sub.empty:
            continue
        y = np.array([order.index(m) for m in sub["source_model"].astype(str).tolist()], dtype=float)
        y = y + offsets[tgt] * spacing
        means = sub["mean_diff_mixed_minus_solo"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        err = np.vstack([means - lo, hi - means])
        ax.errorbar(means, y, xerr=err, fmt="o", ecolor="0.6", elinewidth=0.9,
                    capsize=2, mfc=palette.get(tgt, "#888"), mec="black",
                    mew=0.4, label=tgt, lw=1.0)
    ax.axvline(0.0, linestyle="--", color="0.55", linewidth=1.0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Mean (mixed F1 - best-solo F1) on held-out target")
    ax.set_title("RQ2 forest plot: per-model mixed vs best-solo OOD F1")
    ax.legend(title="target", fontsize=7, loc="best")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, *, decisions: pd.DataFrame, cells: pd.DataFrame,
                  pairs: pd.DataFrame, by_target: pd.DataFrame, decision: bool) -> None:
    n_models = len(decisions)
    n_accept = int(decisions["accept_per_model"].sum()) if not decisions.empty else 0
    fraction = (n_accept / n_models) if n_models else 0.0
    overall = summarize_diffs(pairs["diff_mixed_minus_solo"].to_numpy(dtype=float))

    lines: list[str] = []
    lines.append("# RQ2 - Mixed vs Solo OOD Transfer\n")
    lines.append("## Hypothesis\n")
    lines.append("- Null (H2_0): For each model and held-out target c, the difference "
                 "`mixed_ood_f1(c) - best_solo_ood_f1(c)` has a median equal to zero.")
    lines.append("- Alternative (H2_1): The median difference is positive in favor of "
                 "the mixed regime.\n")
    lines.append("## Acceptance Criteria\n")
    lines.append(f"- Per model: FDR-adjusted p-value < {ALPHA} and Cohen's d >= {RQ2_COHEN_D} "
                 f"on at least {RQ2_TARGETS_REQUIRED} of 3 held-out targets.\n")
    lines.append("## Filter Configuration\n")
    lines.extend(
        describe_filters(
            models_include=MODELS_INCLUDE,
            models_exclude=MODELS_EXCLUDE,
            regimes_include=("solo", "mixed"),
            source_datasets_include=None,
            source_datasets_exclude=(),
            target_datasets_include=TARGET_DATASETS_INCLUDE,
            target_datasets_exclude=TARGET_DATASETS_EXCLUDE,
        )
    )
    lines.append(
        "- Mixed training sources included: "
        f"{'all' if MIXED_SOURCES_INCLUDE is None else ', '.join(MIXED_SOURCES_INCLUDE)}"
    )
    lines.append(
        "- Mixed training sources excluded: "
        f"{'none' if not MIXED_SOURCES_EXCLUDE else ', '.join(MIXED_SOURCES_EXCLUDE)}"
    )
    lines.append(
        "- Solo baseline sources included: "
        f"{'all' if SOLO_SOURCES_INCLUDE is None else ', '.join(SOLO_SOURCES_INCLUDE)}"
    )
    lines.append(
        "- Solo baseline sources excluded: "
        f"{'none' if not SOLO_SOURCES_EXCLUDE else ', '.join(SOLO_SOURCES_EXCLUDE)}\n"
    )
    lines.append("## Global Result\n")
    lines.append(f"- Models tested: {n_models}")
    lines.append(f"- Models meeting criteria: {n_accept} ({fraction:.0%})")
    lines.append(f"- Overall decision: "
                 f"{'ACCEPT H2_1 for majority of models' if decision else 'NOT ACCEPTED FOR MAJORITY'}\n")
    lines.append("## Pooled Effect Across All Pairs\n")
    lines.append(f"- N paired observations: {overall.n}")
    lines.append(f"- Mean diff: {overall.mean:.4f} (95% CI [{overall.ci_low:.4f}, {overall.ci_high:.4f}])")
    lines.append(f"- Median diff: {overall.median:.4f}")
    lines.append(f"- Cohen's d (paired): {overall.cohens_d:.3f}")
    lines.append(f"- Wilcoxon p-value (one-sided, greater): {overall.pvalue:.4g}\n")
    lines.append("## By Held-Out Target\n")
    lines.append(df_to_markdown(by_target))
    lines.append("\n## Per-Model Decisions\n")
    lines.append(df_to_markdown(decisions))
    lines.append("\n## Per-(Model, Target) Test Results\n")
    cols = ["source_model", "target_dataset", "n_folds", "mean_diff_mixed_minus_solo",
            "ci_low", "ci_high", "cohens_d", "p_value_one_sided", "p_value_fdr",
            "cell_supports_h2"]
    lines.append(df_to_markdown(cells[cols]))
    lines.append("\n## Outputs\n")
    lines.append("- `per_pair_observations.csv`: paired mixed and best-solo F1 per fold.")
    lines.append("- `per_model_target_results.csv`: per-(model, target) test statistics.")
    lines.append("- `per_model_decisions.csv`: per-model accept/reject decisions.")
    lines.append("- `by_target.csv`: pooled differences split by held-out target.")
    lines.append("- `forest_plot_rq2.png`: forest plot of per-(model, target) effects.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if df.empty:
        raise RuntimeError("No data after filters; relax MODELS_*/_DATASETS_* settings.")

    pairs = build_pairs(df)
    pairs.to_csv(out_dir / "per_pair_observations.csv", index=False)
    if pairs.empty:
        (out_dir / "decision_summary.md").write_text(
            "# RQ2 - Mixed vs Solo OOD Transfer\n\nNo paired observations under "
            "current filters.\n",
            encoding="utf-8",
        )
        print(f"[OK] RQ2 outputs (empty) in {out_dir}")
        return

    cells = per_cell_results(pairs)
    cells.to_csv(out_dir / "per_model_target_results.csv", index=False)

    decisions = per_model_decision(cells)
    decisions.to_csv(out_dir / "per_model_decisions.csv", index=False)

    by_target = (
        pairs.groupby("target_dataset", as_index=False)
        .agg(
            mean_diff=("diff_mixed_minus_solo", "mean"),
            median_diff=("diff_mixed_minus_solo", "median"),
            n_pairs=("diff_mixed_minus_solo", "count"),
        )
    )
    by_target.to_csv(out_dir / "by_target.csv", index=False)

    plot_forest(cells, out_dir / "forest_plot_rq2.png")

    n_models = len(decisions)
    fraction = (int(decisions["accept_per_model"].sum()) / n_models) if n_models else 0.0
    decision = fraction >= 0.5

    write_summary(out_dir / "decision_summary.md", decisions=decisions, cells=cells,
                  pairs=pairs, by_target=by_target, decision=decision)
    print(f"[OK] RQ2 outputs in {out_dir}")


if __name__ == "__main__":
    main()
