from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


ANALYSIS_DIR = Path("workspace/analysis")
IN_DOMAIN_CSV = ANALYSIS_DIR / "test_eval_results" / "per_checkpoint_test_metrics.csv"
CROSS_CSV = ANALYSIS_DIR / "test_eval_results_cross_dataset" / "cross_dataset_testsplit_metrics.csv"
EFF_CSV = ANALYSIS_DIR / "efficiency_profile" / "per_checkpoint_efficiency.csv"
OUT_DIR = ANALYSIS_DIR / "generalizability"
OUT_MASTER = OUT_DIR / "generalization_master.csv"
OUT_CKPT_SUMMARY = OUT_DIR / "checkpoint_generalization_summary.csv"

CKPT_KEYS = ["cfg_id", "source_model", "source_origin", "source_fold", "best_epoch"]


def _split_components(value: str) -> list[str]:
    if not value:
        return []
    if "," in value:
        return [x.strip() for x in value.split(",") if x.strip()]
    return [x.strip() for x in value.split("_") if x.strip()]


def _in_domain_checkpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["source_model"] = work["model"].astype(str)
    work["source_origin"] = work["origin"].astype(str)
    work["source_fold"] = pd.to_numeric(work["fold"], errors="coerce").astype("Int64")
    work["source_dataset"] = work["train_dataset"].astype(str)
    work["source_dataset_regime"] = work["dataset_regime"].astype(str)
    work["source_dataset_components"] = work["mixed_sources"].astype(str)
    work["in_domain_eval_dataset"] = work["dataset"].astype(str)
    grouped = (
        work.groupby(CKPT_KEYS + ["source_dataset", "source_dataset_regime", "source_dataset_components"], as_index=False)
        .agg(
            in_domain_f1_mean=("test_f1", "mean"),
            in_domain_f1_min=("test_f1", "min"),
            in_domain_acc_mean=("test_acc", "mean"),
            in_domain_acc_min=("test_acc", "min"),
            in_domain_n_eval_datasets=("in_domain_eval_dataset", "nunique"),
            in_domain_eval_datasets=("in_domain_eval_dataset", lambda s: ",".join(sorted(set(s.astype(str))))),
        )
    )
    return grouped


def _efficiency_checkpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["source_model"] = work["model"].astype(str)
    work["source_origin"] = work["origin"].astype(str)
    work["source_fold"] = pd.to_numeric(work["fold"], errors="coerce").astype("Int64")
    grouped = (
        work.groupby(CKPT_KEYS, as_index=False)
        .agg(
            latency_mean_ms=("latency_mean_ms_last_k", "mean"),
            params_count_mean=("params_count", "mean"),
            macs_count_mean=("macs_count", "mean"),
            flops_count_mean=("flops_count", "mean"),
            memory_mean_mb=("memory_mean_mb_last_k", "mean"),
            memory_peak_mb=("memory_peak_mb_max_10_batches", "mean"),
        )
    )
    return grouped


def _cross_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work[work["target_scope"].astype(str) == "test_split"].copy()
    work["source_fold"] = pd.to_numeric(work["source_fold"], errors="coerce").astype("Int64")
    work["source_dataset_components"] = work["source_dataset_components"].astype(str)
    work["target_dataset"] = work["target_dataset"].astype(str)
    return work


def main() -> None:
    in_domain_path = (IN_DOMAIN_CSV if IN_DOMAIN_CSV.is_absolute() else _REPO_ROOT / IN_DOMAIN_CSV).resolve()
    cross_path = (CROSS_CSV if CROSS_CSV.is_absolute() else _REPO_ROOT / CROSS_CSV).resolve()
    eff_path = (EFF_CSV if EFF_CSV.is_absolute() else _REPO_ROOT / EFF_CSV).resolve()

    for p in (in_domain_path, cross_path, eff_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input CSV: {p}")

    in_domain = pd.read_csv(in_domain_path, low_memory=False)
    cross = pd.read_csv(cross_path, low_memory=False)
    eff = pd.read_csv(eff_path, low_memory=False)

    in_ckpt = _in_domain_checkpoint_table(in_domain)
    eff_ckpt = _efficiency_checkpoint_table(eff)
    cross_rows = _cross_table(cross)

    master = cross_rows.merge(
        in_ckpt,
        on=CKPT_KEYS + ["source_dataset", "source_dataset_regime", "source_dataset_components"],
        how="left",
    ).merge(
        eff_ckpt,
        on=CKPT_KEYS,
        how="left",
    )
    if master.empty:
        raise RuntimeError("Generalization master table is empty after joins.")

    components = master["source_dataset_components"].map(_split_components)
    master["n_source_components"] = components.map(len).astype(int)
    master["is_in_domain_target"] = [
        tgt in set(parts) for tgt, parts in zip(master["target_dataset"].astype(str), components, strict=False)
    ]
    master["gap_f1_to_target"] = master["in_domain_f1_mean"] - pd.to_numeric(master["target_f1"], errors="coerce")
    master["source_label"] = master["source_dataset"].astype(str) + " -> " + master["source_model"].astype(str)

    ckpt_summary = (
        master.groupby(CKPT_KEYS + ["source_dataset", "source_dataset_regime"], as_index=False)
        .agg(
            in_domain_f1=("in_domain_f1_mean", "mean"),
            mean_cross_f1=("target_f1", "mean"),
            worst_target_f1=("target_f1", "min"),
            mean_cross_acc=("target_acc", "mean"),
            n_targets=("target_dataset", "nunique"),
            latency_mean_ms=("latency_mean_ms", "mean"),
            params_count_mean=("params_count_mean", "mean"),
            macs_count_mean=("macs_count_mean", "mean"),
        )
    )
    ckpt_summary["gap_f1"] = ckpt_summary["in_domain_f1"] - ckpt_summary["mean_cross_f1"]
    ckpt_summary["robustness_ratio"] = np.where(
        ckpt_summary["mean_cross_f1"] > 0,
        ckpt_summary["worst_target_f1"] / ckpt_summary["mean_cross_f1"],
        np.nan,
    )

    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv((OUT_MASTER if OUT_MASTER.is_absolute() else _REPO_ROOT / OUT_MASTER).resolve(), index=False)
    ckpt_summary.to_csv((OUT_CKPT_SUMMARY if OUT_CKPT_SUMMARY.is_absolute() else _REPO_ROOT / OUT_CKPT_SUMMARY).resolve(), index=False)

    print(f"[OK] wrote {(OUT_MASTER if OUT_MASTER.is_absolute() else _REPO_ROOT / OUT_MASTER).resolve()} ({len(master)} rows)")
    print(f"[OK] wrote {(OUT_CKPT_SUMMARY if OUT_CKPT_SUMMARY.is_absolute() else _REPO_ROOT / OUT_CKPT_SUMMARY).resolve()} ({len(ckpt_summary)} rows)")


if __name__ == "__main__":
    main()
