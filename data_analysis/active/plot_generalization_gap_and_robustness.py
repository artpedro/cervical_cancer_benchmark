from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/analysis/generalizability/plots/gap_robustness")
OUT_SUMMARY_CSV = Path("workspace/analysis/generalizability/generalization_gap_summary.csv")


def _checkpoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(
            ["cfg_id", "source_model", "source_origin", "source_fold", "best_epoch", "source_dataset", "source_dataset_regime"],
            as_index=False,
        )
        .agg(
            in_domain_f1=("in_domain_f1_mean", "mean"),
            mean_cross_f1=("target_f1", "mean"),
            worst_target_f1=("target_f1", "min"),
            n_targets=("target_dataset", "nunique"),
        )
    )
    out["gap_f1"] = out["in_domain_f1"] - out["mean_cross_f1"]
    return out


def _model_regime_summary(ckpt_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        ckpt_df.groupby(["source_dataset_regime", "source_model"], as_index=False)
        .agg(
            in_domain_f1=("in_domain_f1", "mean"),
            mean_cross_f1=("mean_cross_f1", "mean"),
            worst_target_f1=("worst_target_f1", "mean"),
            gap_f1=("gap_f1", "mean"),
            n_checkpoints=("cfg_id", "nunique"),
        )
    )
    out = out.sort_values(["source_dataset_regime", "gap_f1"], ascending=[True, True]).reset_index(drop=True)
    return out


def _plot_ranked_bars(sub: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    if sub.empty:
        return
    d = sub.sort_values(metric, ascending=(metric == "gap_f1")).copy()
    labels = d["source_model"].astype(str).tolist()
    vals = d[metric].astype(float).tolist()
    fig_h = max(4.0, 0.28 * len(labels))
    fig, ax = plt.subplots(figsize=(7.4, fig_h), constrained_layout=True)
    ax.barh(labels, vals)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    master_path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master CSV: {master_path}")
    df = pd.read_csv(master_path, low_memory=False)
    required = {
        "cfg_id",
        "source_model",
        "source_origin",
        "source_fold",
        "best_epoch",
        "source_dataset",
        "source_dataset_regime",
        "in_domain_f1_mean",
        "target_dataset",
        "target_f1",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df["in_domain_f1_mean"] = pd.to_numeric(df["in_domain_f1_mean"], errors="coerce")
    df["target_f1"] = pd.to_numeric(df["target_f1"], errors="coerce")
    df = df.dropna(subset=["in_domain_f1_mean", "target_f1"]).copy()

    ckpt = _checkpoint_summary(df)
    summary = _model_regime_summary(ckpt)

    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for regime in ("solo", "mixed"):
        s = summary[summary["source_dataset_regime"].astype(str) == regime].copy()
        if s.empty:
            continue
        _plot_ranked_bars(
            s,
            metric="gap_f1",
            title=f"Generalization Gap (in-domain F1 - cross-domain F1) [{regime}]",
            out_path=out_dir / f"ranked_gap_f1_{regime}.png",
        )
        _plot_ranked_bars(
            s,
            metric="worst_target_f1",
            title=f"Worst-target F1 robustness [{regime}]",
            out_path=out_dir / f"ranked_worst_target_f1_{regime}.png",
        )

    summary_csv = (OUT_SUMMARY_CSV if OUT_SUMMARY_CSV.is_absolute() else _REPO_ROOT / OUT_SUMMARY_CSV).resolve()
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] wrote {summary_csv}")


if __name__ == "__main__":
    main()
