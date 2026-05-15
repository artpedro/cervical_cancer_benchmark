"""Statistical analysis for RQ5.

RQ5: Efficiency vs Generalizability Pareto Front.

- Null (H5_0): Kendall tau between in-domain F1 ranks and OOD F1 ranks of
  models is one (perfect agreement).
- Alternative (H5_1): Kendall tau is significantly less than one, indicating
  that model rankings (and thus the Pareto front) differ between the two
  views.

Analyses produced:
1. Per-model means: in-domain F1, OOD F1, latency, params, MACs.
2. Kendall tau between in-domain and OOD F1 rankings, computed overall and
   per regime, with bootstrap 95% CIs.
3. Pareto front computation for (latency vs F1) using both in-domain F1 and
   OOD F1; rank changes per model are reported.
4. Side-by-side Pareto plot.

Outputs (workspace/statistical_results/RQ5):
- decision_summary.md
- per_model_means.csv
- kendall_tau_results.csv
- pareto_front_membership.csv
- rank_changes.csv
- pareto_plot_rq5.png

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
    bootstrap_ci_paired_statistic,
    describe_filters,
    df_to_markdown,
    model_family,
)


# =============================================================================
# CONSTANTS - edit to filter models, regimes, or dataset configurations
# =============================================================================

MODELS_INCLUDE: tuple[str, ...] | None = None
MODELS_EXCLUDE: tuple[str, ...] = ('EAT')

# Regimes for which Kendall tau and Pareto fronts are computed. Each regime
# in this list produces its own ranking and Pareto front pair.
REGIMES_INCLUDE: tuple[str, ...] = ("solo", "mixed")

SOURCE_DATASETS_INCLUDE: tuple[str, ...] | None = None
SOURCE_DATASETS_EXCLUDE: tuple[str, ...] = ()

TARGET_DATASETS_INCLUDE: tuple[str, ...] | None = None
TARGET_DATASETS_EXCLUDE: tuple[str, ...] = ()

# Acceptance criteria for H5_1.
ALPHA: float = 0.05
KENDALL_TAU_THRESHOLD: float = 0.85
N_BOOTSTRAP: int = 10_000
BOOTSTRAP_SEED: int = 42

# Cost axis for the Pareto front (must exist in master CSV).
COST_COLUMN: str = "latency_mean_ms"

# =============================================================================

MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/statistical_results/RQ5")


def load_data() -> pd.DataFrame:
    path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing master CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in ("target_f1", "in_domain_f1_mean", COST_COLUMN, "params_count_mean", "macs_count_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target_f1", "in_domain_f1_mean", COST_COLUMN]).copy()
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


def per_model_means(df: pd.DataFrame, regime: str | None) -> pd.DataFrame:
    sub = df if regime is None else df[df["source_dataset_regime"].astype(str) == regime]
    if sub.empty:
        return pd.DataFrame()
    grouped = (
        sub.groupby("source_model", as_index=False)
        .agg(
            in_domain_f1=("in_domain_f1_mean", "mean"),
            ood_f1=("target_f1", "mean"),
            cost=(COST_COLUMN, "mean"),
            params=("params_count_mean", "mean"),
            macs=("macs_count_mean", "mean"),
            n_checkpoints=("cfg_id", "nunique"),
        )
    )
    grouped["family"] = grouped["source_model"].astype(str).map(model_family)
    return grouped


def kendall_with_ci(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size < 3 or y.size < 3:
        return {"tau": float("nan"), "p_two_sided": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_models": int(x.size)}
    res = stats.kendalltau(x, y)
    tau = float(res.correlation)
    p = float(res.pvalue)

    def _stat(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3:
            return float("nan")
        r = stats.kendalltau(a, b)
        return float(r.correlation)

    ci_lo, ci_hi = bootstrap_ci_paired_statistic(
        x, y, _stat, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    return {"tau": tau, "p_two_sided": p, "ci_low": ci_lo, "ci_high": ci_hi,
            "n_models": int(x.size)}


def pareto_front(cost: np.ndarray, performance: np.ndarray) -> np.ndarray:
    """Return boolean mask of points on the (min cost, max performance) front."""
    n = cost.size
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (cost[j] <= cost[i] and performance[j] >= performance[i]
                    and (cost[j] < cost[i] or performance[j] > performance[i])):
                on_front[i] = False
                break
    return on_front


def rank_changes(per_model: pd.DataFrame) -> pd.DataFrame:
    if per_model.empty:
        return pd.DataFrame()
    out = per_model.copy()
    out["in_domain_rank"] = (
        out["in_domain_f1"].rank(method="min", ascending=False).astype(int)
    )
    out["ood_rank"] = out["ood_f1"].rank(method="min", ascending=False).astype(int)
    out["rank_delta"] = out["ood_rank"] - out["in_domain_rank"]
    return out.sort_values("rank_delta", ascending=False).reset_index(drop=True)


def compute_pareto_membership(per_model: pd.DataFrame) -> pd.DataFrame:
    if per_model.empty:
        return pd.DataFrame()
    cost = per_model["cost"].to_numpy(dtype=float)
    in_perf = per_model["in_domain_f1"].to_numpy(dtype=float)
    ood_perf = per_model["ood_f1"].to_numpy(dtype=float)
    in_front = pareto_front(cost, in_perf)
    ood_front = pareto_front(cost, ood_perf)
    out = per_model[["source_model", "family", "cost", "in_domain_f1", "ood_f1"]].copy()
    out["on_in_domain_front"] = in_front
    out["on_ood_front"] = ood_front
    out["membership_change"] = np.where(
        in_front == ood_front, "stable",
        np.where(ood_front, "added_in_ood", "removed_in_ood"),
    )
    return out.reset_index(drop=True)


def plot_pareto(per_model: pd.DataFrame, regime: str, out_path: Path) -> None:
    if per_model.empty:
        return
    cost = per_model["cost"].to_numpy(dtype=float)
    in_f1 = per_model["in_domain_f1"].to_numpy(dtype=float)
    ood_f1 = per_model["ood_f1"].to_numpy(dtype=float)
    in_mask = pareto_front(cost, in_f1)
    ood_mask = pareto_front(cost, ood_f1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True, sharey=True)
    titles = ["In-domain F1", "OOD F1"]
    perfs = [in_f1, ood_f1]
    masks = [in_mask, ood_mask]

    palette = {"transformer": "#c44e52", "cnn": "#3a76af"}
    for ax, title, perf, mask in zip(axes, titles, perfs, masks):
        for fam in ("cnn", "transformer"):
            sel = (per_model["family"].astype(str) == fam).to_numpy()
            ax.scatter(cost[sel], perf[sel], s=32, c=palette.get(fam, "#777"),
                       label=fam, alpha=0.85, edgecolor="white", linewidth=0.4)
        front = np.where(mask)[0]
        order = np.argsort(cost[front])
        if order.size:
            ax.plot(cost[front][order], perf[front][order], color="#2a9134",
                    linewidth=1.5, alpha=0.85, label="pareto")
            ax.scatter(cost[front], perf[front], s=72, facecolor="none", edgecolor="#2a9134",
                       linewidth=1.0, zorder=4)
        ax.set_xlabel(f"{COST_COLUMN}")
        ax.set_title(f"{title} ({regime})")
        ax.grid(linestyle="--", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("F1")
            ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, *, regime_results: dict[str, dict],
                  decision: bool) -> None:
    lines: list[str] = []
    lines.append("# RQ5 - Efficiency vs Generalizability Ranking Stability\n")
    lines.append("## Hypothesis\n")
    lines.append("- Null (H5_0): Kendall tau between in-domain F1 ranks and OOD F1 "
                 "ranks of models is one (perfect agreement).")
    lines.append("- Alternative (H5_1): Kendall tau is below one, so rankings and "
                 "Pareto fronts differ between in-domain and OOD views.\n")
    lines.append("## Acceptance Criteria\n")
    lines.append(f"- For at least one regime: Kendall tau < {KENDALL_TAU_THRESHOLD}, AND")
    lines.append("- Bootstrap 95% CI on tau excludes 1.0.\n")
    lines.append("## Configuration\n")
    lines.append(f"- Cost axis for Pareto: {COST_COLUMN}")
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
    lines.append(f"- Overall decision: {'ACCEPT H5_1' if decision else 'REJECT H5_1'}\n")
    lines.append("## Per-Regime Kendall Tau\n")
    rows = []
    for regime, info in regime_results.items():
        rows.append({
            "regime": regime,
            "n_models": info["tau"]["n_models"],
            "kendall_tau": info["tau"]["tau"],
            "p_two_sided": info["tau"]["p_two_sided"],
            "ci_low": info["tau"]["ci_low"],
            "ci_high": info["tau"]["ci_high"],
            "models_added_in_ood_pareto": int(info["pareto_added"]),
            "models_removed_in_ood_pareto": int(info["pareto_removed"]),
            "max_rank_delta": int(info["max_rank_delta"]),
        })
    lines.append(df_to_markdown(pd.DataFrame(rows)))
    for regime, info in regime_results.items():
        lines.append(f"\n## Rank Changes ({regime})\n")
        lines.append(df_to_markdown(info["rank_changes"][[
            "source_model", "family", "in_domain_f1", "ood_f1",
            "in_domain_rank", "ood_rank", "rank_delta"
        ]]))
        lines.append(f"\n## Pareto Membership ({regime})\n")
        lines.append(df_to_markdown(info["pareto_membership"]))
    lines.append("\n## Outputs\n")
    lines.append("- `per_model_means.csv`: per-model means used for ranking and Pareto.")
    lines.append("- `kendall_tau_results.csv`: Kendall tau with bootstrap CIs per regime.")
    lines.append("- `rank_changes.csv`: rank deltas between in-domain and OOD per regime.")
    lines.append("- `pareto_front_membership.csv`: per-model Pareto membership and changes.")
    lines.append("- `pareto_plot_rq5.png`: side-by-side Pareto plots per regime.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if df.empty:
        raise RuntimeError("No data after filtering for RQ5.")

    per_model_all_frames: list[pd.DataFrame] = []
    rank_change_frames: list[pd.DataFrame] = []
    pareto_frames: list[pd.DataFrame] = []
    regime_results: dict[str, dict] = {}

    for regime in REGIMES_INCLUDE:
        per_model = per_model_means(df, regime)
        if per_model.empty:
            continue
        per_model = per_model.copy()
        per_model["regime"] = regime
        per_model_all_frames.append(per_model)

        x = per_model["in_domain_f1"].to_numpy(dtype=float)
        y = per_model["ood_f1"].to_numpy(dtype=float)
        tau = kendall_with_ci(x, y)

        rc = rank_changes(per_model.drop(columns=["regime"]))
        rc["regime"] = regime
        rank_change_frames.append(rc)

        pareto = compute_pareto_membership(per_model.drop(columns=["regime"]))
        pareto["regime"] = regime
        pareto_frames.append(pareto)

        plot_pareto(per_model, regime, out_dir / f"pareto_plot_rq5_{regime}.png")

        regime_results[regime] = {
            "tau": tau,
            "rank_changes": rc,
            "pareto_membership": pareto,
            "pareto_added": int((pareto["membership_change"] == "added_in_ood").sum()),
            "pareto_removed": int((pareto["membership_change"] == "removed_in_ood").sum()),
            "max_rank_delta": int(np.abs(rc["rank_delta"]).max() if not rc.empty else 0),
        }

    if not regime_results:
        raise RuntimeError("No regime produced rankings; check filters.")

    pd.concat(per_model_all_frames, ignore_index=True).to_csv(
        out_dir / "per_model_means.csv", index=False
    )
    pd.concat(rank_change_frames, ignore_index=True).to_csv(
        out_dir / "rank_changes.csv", index=False
    )
    pd.concat(pareto_frames, ignore_index=True).to_csv(
        out_dir / "pareto_front_membership.csv", index=False
    )
    tau_rows = []
    for regime, info in regime_results.items():
        tau_rows.append({"regime": regime, **info["tau"]})
    pd.DataFrame(tau_rows).to_csv(out_dir / "kendall_tau_results.csv", index=False)

    decision = any(
        info["tau"]["tau"] is not None
        and not np.isnan(info["tau"]["tau"])
        and info["tau"]["tau"] < KENDALL_TAU_THRESHOLD
        and (np.isnan(info["tau"]["ci_high"]) or info["tau"]["ci_high"] < 1.0)
        for info in regime_results.values()
    )

    write_summary(out_dir / "decision_summary.md", regime_results=regime_results,
                  decision=decision)
    print(f"[OK] RQ5 outputs in {out_dir}")


if __name__ == "__main__":
    main()
