from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


CFG_KEYS = ["dataset", "model", "origin", "fold"]
RUN_ID_KEYS = [
    "source_metrics_root",
    "source_dataset_path",
    "source_balance_mode_path",
    "source_run_timestamp_path",
]


def _latest_all_metrics_dir(workspace_dir: Path) -> Path:
    candidates = sorted(
        p / "all_metrics"
        for p in workspace_dir.glob("all_metrics_copy_*")
        if (p / "all_metrics").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No all_metrics directories found under: {workspace_dir}"
        )
    return candidates[-1]


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _pick_best_summary_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(summary_df, CFG_KEYS + RUN_ID_KEYS + ["best_acc", "best_f1"], "summary")
    work = _coerce_numeric(summary_df, ["fold", "best_acc", "best_f1"]).copy()
    work["source_run_timestamp_path"] = work["source_run_timestamp_path"].fillna("").astype(str)
    work["source_csv_relpath"] = work.get("source_csv_relpath", "").fillna("").astype(str)

    # Sort descending by best metrics; newest timestamp as additional tie-break.
    work = work.sort_values(
        by=CFG_KEYS + ["best_acc", "best_f1", "source_run_timestamp_path", "source_csv_relpath"],
        ascending=[True, True, True, True, False, False, False, False],
        na_position="last",
    )
    # keep first row per configuration after sorting
    best = work.drop_duplicates(subset=CFG_KEYS, keep="first").reset_index(drop=True)
    return best


def _filter_by_winning_runs(df: pd.DataFrame, winners: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    _require_cols(df, CFG_KEYS + RUN_ID_KEYS, name)
    left = df.copy()
    right = winners[CFG_KEYS + RUN_ID_KEYS].drop_duplicates().copy()

    # Normalize dtypes for safe merge
    left["fold"] = pd.to_numeric(left["fold"], errors="coerce")
    right["fold"] = pd.to_numeric(right["fold"], errors="coerce")
    for c in ["dataset", "model", "origin"] + RUN_ID_KEYS:
        left[c] = left[c].fillna("").astype(str)
        right[c] = right[c].fillna("").astype(str)

    merged = left.merge(
        right,
        on=CFG_KEYS + RUN_ID_KEYS,
        how="inner",
    )
    return merged


def _dedup_training_time_fallback(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback if no run-id-aligned rows were found: keep the best record per config
    based on best_acc, then best_f1.
    """
    if training_df.empty:
        return training_df.copy()

    required = CFG_KEYS + ["best_acc", "best_f1"]
    _require_cols(training_df, required, "training_time_results")
    work = _coerce_numeric(training_df, ["fold", "best_acc", "best_f1"]).copy()
    work = work.sort_values(
        by=CFG_KEYS + ["best_acc", "best_f1"],
        ascending=[True, True, True, True, False, False],
        na_position="last",
    )
    return work.drop_duplicates(subset=CFG_KEYS, keep="first").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate merged all_metrics CSVs by keeping best run per "
            "(dataset, model, origin, fold) based on best summary metrics."
        )
    )
    parser.add_argument(
        "--all-metrics-dir",
        type=Path,
        default=None,
        help=(
            "Path to all_metrics folder (contains all_*.csv). "
            "If omitted, uses latest workspace/all_metrics_copy_*/all_metrics."
        ),
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("workspace"),
        help="Workspace root used when --all-metrics-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for deduplicated CSVs. "
            "Default: <all-metrics-dir>/deduplicated_best"
        ),
    )
    args = parser.parse_args()

    all_metrics_dir = (
        args.all_metrics_dir.resolve()
        if args.all_metrics_dir is not None
        else _latest_all_metrics_dir(args.workspace_dir.resolve())
    )
    if not all_metrics_dir.exists():
        raise FileNotFoundError(f"all_metrics directory does not exist: {all_metrics_dir}")

    summary_path = all_metrics_dir / "all_summary_weighted_loss.csv"
    epoch_logs_path = all_metrics_dir / "all_epoch_logs_weighted_loss.csv"
    fold_epoch_path = all_metrics_dir / "all_fold_epoch_metrics.csv"
    training_time_path = all_metrics_dir / "all_training_time_results.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required file: {summary_path}")

    summary_df = pd.read_csv(summary_path, low_memory=False)
    epoch_logs_df = pd.read_csv(epoch_logs_path, low_memory=False) if epoch_logs_path.exists() else pd.DataFrame()
    fold_epoch_df = pd.read_csv(fold_epoch_path, low_memory=False) if fold_epoch_path.exists() else pd.DataFrame()
    training_time_df = (
        pd.read_csv(training_time_path, low_memory=False)
        if training_time_path.exists()
        else pd.DataFrame()
    )

    winners = _pick_best_summary_rows(summary_df)

    summary_dedup = winners.copy()
    epoch_logs_dedup = _filter_by_winning_runs(epoch_logs_df, winners, "epoch_logs")
    fold_epoch_dedup = _filter_by_winning_runs(fold_epoch_df, winners, "fold_epoch")

    training_time_dedup = _filter_by_winning_runs(
        training_time_df, winners, "training_time_results"
    )
    used_training_fallback = False
    if training_time_df.shape[0] > 0 and training_time_dedup.shape[0] == 0:
        training_time_dedup = _dedup_training_time_fallback(training_time_df)
        used_training_fallback = True

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (all_metrics_dir / "deduplicated_best").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary = out_dir / "all_summary_weighted_loss.dedup_best.csv"
    out_epoch_logs = out_dir / "all_epoch_logs_weighted_loss.dedup_best.csv"
    out_fold_epoch = out_dir / "all_fold_epoch_metrics.dedup_best.csv"
    out_training = out_dir / "all_training_time_results.dedup_best.csv"

    summary_dedup.to_csv(out_summary, index=False)
    epoch_logs_dedup.to_csv(out_epoch_logs, index=False)
    fold_epoch_dedup.to_csv(out_fold_epoch, index=False)
    training_time_dedup.to_csv(out_training, index=False)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "all_metrics_dir": str(all_metrics_dir),
        "output_dir": str(out_dir),
        "rule": "keep max(best_acc), tie-break max(best_f1), then latest source_run_timestamp_path",
        "rows_before": {
            "summary": int(len(summary_df)),
            "epoch_logs": int(len(epoch_logs_df)),
            "fold_epoch_metrics": int(len(fold_epoch_df)),
            "training_time_results": int(len(training_time_df)),
        },
        "rows_after": {
            "summary": int(len(summary_dedup)),
            "epoch_logs": int(len(epoch_logs_dedup)),
            "fold_epoch_metrics": int(len(fold_epoch_dedup)),
            "training_time_results": int(len(training_time_dedup)),
        },
        "unique_configs_after": int(summary_dedup[CFG_KEYS].drop_duplicates().shape[0]),
        "training_time_used_fallback": used_training_fallback,
    }
    (out_dir / "dedup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Wrote deduplicated CSVs to: {out_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

