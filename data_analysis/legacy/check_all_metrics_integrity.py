from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


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


def _read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mask = df.duplicated(keep=False)
    return df[mask].copy()


def _key_duplicates(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    for k in keys:
        if k not in df.columns:
            return pd.DataFrame()
    counts = (
        df.groupby(keys, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )
    return counts[counts["row_count"] > 1].reset_index(drop=True)


@dataclass(frozen=True)
class CoverageIssue:
    issue_type: str
    dataset: str
    model: str
    origin: str
    fold: str
    details: str


def _coverage_checks(
    summary_df: pd.DataFrame,
    epoch_logs_df: pd.DataFrame,
    expected_folds: int,
) -> list[CoverageIssue]:
    issues: list[CoverageIssue] = []

    if summary_df.empty:
        issues.append(
            CoverageIssue(
                issue_type="missing_file",
                dataset="",
                model="",
                origin="",
                fold="",
                details="all_summary_weighted_loss.csv is missing or empty",
            )
        )
        return issues

    req_cols = {"dataset", "model", "origin", "fold", "best_epoch"}
    if not req_cols.issubset(summary_df.columns):
        issues.append(
            CoverageIssue(
                issue_type="schema_error",
                dataset="",
                model="",
                origin="",
                fold="",
                details=(
                    "summary file missing required columns: "
                    f"{sorted(req_cols - set(summary_df.columns))}"
                ),
            )
        )
        return issues

    # 1) Each (dataset, model, origin) should have all folds [0..expected_folds-1]
    expected_fold_set = set(range(expected_folds))
    grouped = summary_df.groupby(["dataset", "model", "origin"], dropna=False)
    for (dataset, model, origin), g in grouped:
        folds = set(int(x) for x in pd.to_numeric(g["fold"], errors="coerce").dropna())
        missing = sorted(expected_fold_set - folds)
        extra = sorted(folds - expected_fold_set)
        if missing:
            issues.append(
                CoverageIssue(
                    issue_type="missing_folds",
                    dataset=str(dataset),
                    model=str(model),
                    origin=str(origin),
                    fold="",
                    details=f"Missing folds: {missing}",
                )
            )
        if extra:
            issues.append(
                CoverageIssue(
                    issue_type="unexpected_folds",
                    dataset=str(dataset),
                    model=str(model),
                    origin=str(origin),
                    fold="",
                    details=f"Unexpected folds: {extra}",
                )
            )

    # 2) Best epoch in summary must exist in epoch_logs for split=val and split=train
    if epoch_logs_df.empty:
        issues.append(
            CoverageIssue(
                issue_type="missing_file",
                dataset="",
                model="",
                origin="",
                fold="",
                details="all_epoch_logs_weighted_loss.csv is missing or empty",
            )
        )
        return issues

    ereq_cols = {"dataset", "model", "origin", "fold", "epoch", "split"}
    if not ereq_cols.issubset(epoch_logs_df.columns):
        issues.append(
            CoverageIssue(
                issue_type="schema_error",
                dataset="",
                model="",
                origin="",
                fold="",
                details=(
                    "epoch_logs file missing required columns: "
                    f"{sorted(ereq_cols - set(epoch_logs_df.columns))}"
                ),
            )
        )
        return issues

    lookup = epoch_logs_df[
        ["dataset", "model", "origin", "fold", "epoch", "split"]
    ].copy()
    lookup["fold"] = pd.to_numeric(lookup["fold"], errors="coerce")
    lookup["epoch"] = pd.to_numeric(lookup["epoch"], errors="coerce")
    lookup = lookup.dropna(subset=["fold", "epoch"])

    val_idx = set(
        zip(
            lookup.loc[lookup["split"] == "val", "dataset"].astype(str),
            lookup.loc[lookup["split"] == "val", "model"].astype(str),
            lookup.loc[lookup["split"] == "val", "origin"].astype(str),
            lookup.loc[lookup["split"] == "val", "fold"].astype(int),
            lookup.loc[lookup["split"] == "val", "epoch"].astype(int),
        )
    )
    train_idx = set(
        zip(
            lookup.loc[lookup["split"] == "train", "dataset"].astype(str),
            lookup.loc[lookup["split"] == "train", "model"].astype(str),
            lookup.loc[lookup["split"] == "train", "origin"].astype(str),
            lookup.loc[lookup["split"] == "train", "fold"].astype(int),
            lookup.loc[lookup["split"] == "train", "epoch"].astype(int),
        )
    )

    for _, row in summary_df.iterrows():
        dataset = str(row["dataset"])
        model = str(row["model"])
        origin = str(row["origin"])
        fold = int(row["fold"])
        best_epoch = int(row["best_epoch"])
        key = (dataset, model, origin, fold, best_epoch)
        if key not in val_idx:
            issues.append(
                CoverageIssue(
                    issue_type="missing_best_epoch_val_row",
                    dataset=dataset,
                    model=model,
                    origin=origin,
                    fold=str(fold),
                    details=f"epoch_logs missing val row at best_epoch={best_epoch}",
                )
            )
        if key not in train_idx:
            issues.append(
                CoverageIssue(
                    issue_type="missing_best_epoch_train_row",
                    dataset=dataset,
                    model=model,
                    origin=origin,
                    fold=str(fold),
                    details=f"epoch_logs missing train row at best_epoch={best_epoch}",
                )
            )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check merged all_metrics for duplicates and training coverage completeness."
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
        help="Workspace root (used only when --all-metrics-dir is omitted).",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=5,
        help="Expected fold count per (dataset, model, origin) configuration.",
    )
    args = parser.parse_args()

    all_metrics_dir = (
        args.all_metrics_dir.resolve()
        if args.all_metrics_dir is not None
        else _latest_all_metrics_dir(args.workspace_dir.resolve())
    )

    summary_path = all_metrics_dir / "all_summary_weighted_loss.csv"
    epoch_logs_path = all_metrics_dir / "all_epoch_logs_weighted_loss.csv"
    fold_epoch_path = all_metrics_dir / "all_fold_epoch_metrics.csv"
    training_time_path = all_metrics_dir / "all_training_time_results.csv"

    summary_df = _read_if_exists(summary_path)
    epoch_logs_df = _read_if_exists(epoch_logs_path)
    fold_epoch_df = _read_if_exists(fold_epoch_path)
    training_time_df = _read_if_exists(training_time_path)

    report_dir = all_metrics_dir / "integrity_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Duplicate checks
    summary_exact_dup = _exact_duplicates(summary_df)
    epoch_logs_exact_dup = _exact_duplicates(epoch_logs_df)
    fold_epoch_exact_dup = _exact_duplicates(fold_epoch_df)
    training_time_exact_dup = _exact_duplicates(training_time_df)

    summary_key_dup = _key_duplicates(
        summary_df, ["dataset", "model", "origin", "fold"]
    )
    epoch_logs_key_dup = _key_duplicates(
        epoch_logs_df, ["dataset", "model", "origin", "fold", "epoch", "split"]
    )
    fold_epoch_key_dup = _key_duplicates(
        fold_epoch_df, ["dataset", "model", "origin", "fold", "epoch", "split"]
    )
    training_time_key_dup = _key_duplicates(
        training_time_df, ["dataset", "model", "origin", "fold"]
    )

    summary_exact_dup.to_csv(report_dir / "duplicates_summary_exact_rows.csv", index=False)
    epoch_logs_exact_dup.to_csv(
        report_dir / "duplicates_epoch_logs_exact_rows.csv", index=False
    )
    fold_epoch_exact_dup.to_csv(
        report_dir / "duplicates_fold_epoch_exact_rows.csv", index=False
    )
    training_time_exact_dup.to_csv(
        report_dir / "duplicates_training_time_exact_rows.csv", index=False
    )

    summary_key_dup.to_csv(report_dir / "duplicates_summary_by_key.csv", index=False)
    epoch_logs_key_dup.to_csv(report_dir / "duplicates_epoch_logs_by_key.csv", index=False)
    fold_epoch_key_dup.to_csv(report_dir / "duplicates_fold_epoch_by_key.csv", index=False)
    training_time_key_dup.to_csv(
        report_dir / "duplicates_training_time_by_key.csv", index=False
    )

    # Coverage checks
    issues = _coverage_checks(
        summary_df=summary_df,
        epoch_logs_df=epoch_logs_df,
        expected_folds=args.expected_folds,
    )
    issues_df = pd.DataFrame([i.__dict__ for i in issues])
    issues_df.to_csv(report_dir / "coverage_issues.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "all_metrics_dir": str(all_metrics_dir),
        "expected_folds": args.expected_folds,
        "files_present": {
            "all_summary_weighted_loss.csv": summary_path.exists(),
            "all_epoch_logs_weighted_loss.csv": epoch_logs_path.exists(),
            "all_fold_epoch_metrics.csv": fold_epoch_path.exists(),
            "all_training_time_results.csv": training_time_path.exists(),
        },
        "rows": {
            "summary": int(len(summary_df)),
            "epoch_logs": int(len(epoch_logs_df)),
            "fold_epoch_metrics": int(len(fold_epoch_df)),
            "training_time_results": int(len(training_time_df)),
        },
        "duplicate_counts": {
            "summary_exact_rows": int(len(summary_exact_dup)),
            "epoch_logs_exact_rows": int(len(epoch_logs_exact_dup)),
            "fold_epoch_exact_rows": int(len(fold_epoch_exact_dup)),
            "training_time_exact_rows": int(len(training_time_exact_dup)),
            "summary_by_key": int(len(summary_key_dup)),
            "epoch_logs_by_key": int(len(epoch_logs_key_dup)),
            "fold_epoch_by_key": int(len(fold_epoch_key_dup)),
            "training_time_by_key": int(len(training_time_key_dup)),
        },
        "coverage_issue_count": int(len(issues)),
        "report_dir": str(report_dir),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Integrity report written to: {report_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

