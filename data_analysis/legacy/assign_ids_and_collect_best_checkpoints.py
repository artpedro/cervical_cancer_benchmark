from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


CFG_KEYS = ["dataset", "model", "origin", "fold"]
RUN_KEYS = [
    "source_metrics_root",
    "source_dataset_path",
    "source_balance_mode_path",
    "source_run_timestamp_path",
]


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


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
        raise ValueError(f"{name} is missing required columns: {missing}")


def _assign_mds_ids(epoch_logs_dedup: pd.DataFrame, summary_dedup: pd.DataFrame) -> pd.DataFrame:
    # Base combos from epoch logs
    combos = (
        epoch_logs_dedup[["dataset", "model", "split"]]
        .drop_duplicates()
        .sort_values(["dataset", "model", "split"])
        .reset_index(drop=True)
    )

    # Ensure "test" split id exists for every dataset+model in summary
    dm = (
        summary_dedup[["dataset", "model"]]
        .drop_duplicates()
        .assign(split="test")
    )
    combos = (
        pd.concat([combos, dm], axis=0, ignore_index=True)
        .drop_duplicates()
        .sort_values(["dataset", "model", "split"])
        .reset_index(drop=True)
    )

    combos["mds_id"] = [f"MDS{i:04d}" for i in range(1, len(combos) + 1)]
    return combos[["mds_id", "dataset", "model", "split"]]


def _assign_cfg_ids(summary_dedup: pd.DataFrame) -> pd.DataFrame:
    cfg = (
        summary_dedup[CFG_KEYS]
        .copy()
        .drop_duplicates()
    )
    cfg["fold"] = pd.to_numeric(cfg["fold"], errors="coerce").astype("Int64")
    cfg = cfg.sort_values(["dataset", "model", "origin", "fold"]).reset_index(drop=True)
    cfg["cfg_id"] = [f"CFG{i:04d}" for i in range(1, len(cfg) + 1)]
    return cfg[["cfg_id"] + CFG_KEYS]


def _resolve_ckpt_path(all_metrics_dir: Path, row: pd.Series) -> Path:
    copied_metrics_root = all_metrics_dir.parent / "copied_metrics"
    fold_int = int(row["fold"])
    return (
        copied_metrics_root
        / str(row["source_metrics_root"])
        / str(row["source_dataset_path"])
        / str(row["source_balance_mode_path"])
        / str(row["source_run_timestamp_path"])
        / slugify(str(row["model"]))
        / f"fold_{fold_int:02d}"
        / "checkpoints"
        / "best.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assign model+dataset+split IDs and copy best checkpoints for deduplicated runs."
        )
    )
    parser.add_argument(
        "--all-metrics-dir",
        type=Path,
        default=None,
        help=(
            "Path to all_metrics folder. If omitted, uses latest "
            "workspace/all_metrics_copy_*/all_metrics."
        ),
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("workspace"),
        help="Workspace root used when --all-metrics-dir is omitted.",
    )
    args = parser.parse_args()

    all_metrics_dir = (
        args.all_metrics_dir.resolve()
        if args.all_metrics_dir is not None
        else _latest_all_metrics_dir(args.workspace_dir.resolve())
    )
    dedup_dir = all_metrics_dir / "deduplicated_best"
    if not dedup_dir.exists():
        raise FileNotFoundError(
            f"Missing deduplicated directory: {dedup_dir}\n"
            "Run deduplicate_all_metrics_best.py first."
        )

    summary_path = dedup_dir / "all_summary_weighted_loss.dedup_best.csv"
    epoch_logs_path = dedup_dir / "all_epoch_logs_weighted_loss.dedup_best.csv"
    if not summary_path.exists() or not epoch_logs_path.exists():
        raise FileNotFoundError(
            "Missing required deduplicated files:\n"
            f"- {summary_path}\n"
            f"- {epoch_logs_path}"
        )

    summary_df = pd.read_csv(summary_path, low_memory=False)
    epoch_logs_df = pd.read_csv(epoch_logs_path, low_memory=False)

    _require_cols(summary_df, CFG_KEYS + RUN_KEYS + ["best_epoch"], "summary_dedup")
    _require_cols(epoch_logs_df, ["dataset", "model", "split"], "epoch_logs_dedup")

    summary_df["fold"] = pd.to_numeric(summary_df["fold"], errors="coerce")
    summary_df["best_epoch"] = pd.to_numeric(summary_df["best_epoch"], errors="coerce")
    summary_df = summary_df.dropna(subset=["fold", "best_epoch"]).copy()
    summary_df["fold"] = summary_df["fold"].astype(int)
    summary_df["best_epoch"] = summary_df["best_epoch"].astype(int)

    mds_ids = _assign_mds_ids(epoch_logs_df, summary_df)
    cfg_ids = _assign_cfg_ids(summary_df)

    # Attach IDs to summary/checkpoint records
    summary_with_cfg = summary_df.merge(cfg_ids, on=CFG_KEYS, how="left")
    summary_with_ids = summary_with_cfg.merge(
        mds_ids.rename(columns={"split": "_join_split"}).query("_join_split == 'val'")[["mds_id", "dataset", "model"]],
        on=["dataset", "model"],
        how="left",
    )
    summary_with_ids = summary_with_ids.rename(columns={"mds_id": "mds_id_val"})

    # Build checkpoint copy manifest and copy files
    checkpoints_dir = all_metrics_dir / "best_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []

    for _, row in summary_with_ids.iterrows():
        src_ckpt = _resolve_ckpt_path(all_metrics_dir, row)
        cfg_id = str(row["cfg_id"])
        mds_val = str(row.get("mds_id_val", ""))
        dataset = str(row["dataset"])
        model_slug = slugify(str(row["model"]))
        fold = int(row["fold"])
        best_epoch = int(row["best_epoch"])
        dst_name = (
            f"{cfg_id}__{mds_val}__{dataset}__{model_slug}__fold_{fold:02d}__epoch_{best_epoch}.pt"
        )
        dst_ckpt = checkpoints_dir / dst_name

        rec = {
            "cfg_id": cfg_id,
            "mds_id_val": mds_val,
            "dataset": dataset,
            "model": str(row["model"]),
            "origin": str(row["origin"]),
            "fold": fold,
            "best_epoch": best_epoch,
            "source_checkpoint_path": str(src_ckpt),
            "copied_checkpoint_path": str(dst_ckpt),
        }

        if src_ckpt.exists():
            shutil.copy2(src_ckpt, dst_ckpt)
            rec["status"] = "copied"
            copied.append(rec)
        else:
            rec["status"] = "missing_source"
            missing.append(rec)

    copied_df = pd.DataFrame(copied)
    missing_df = pd.DataFrame(missing)
    copied_df.to_csv(all_metrics_dir / "best_checkpoints_index.csv", index=False)
    missing_df.to_csv(all_metrics_dir / "best_checkpoints_missing.csv", index=False)
    mds_ids.to_csv(all_metrics_dir / "model_dataset_split_ids.csv", index=False)
    cfg_ids.to_csv(all_metrics_dir / "config_ids.csv", index=False)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "all_metrics_dir": str(all_metrics_dir),
        "deduplicated_dir": str(dedup_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "counts": {
            "mds_ids": int(len(mds_ids)),
            "cfg_ids": int(len(cfg_ids)),
            "checkpoints_copied": int(len(copied_df)),
            "checkpoints_missing": int(len(missing_df)),
        },
        "outputs": {
            "model_dataset_split_ids": str(all_metrics_dir / "model_dataset_split_ids.csv"),
            "config_ids": str(all_metrics_dir / "config_ids.csv"),
            "best_checkpoints_index": str(all_metrics_dir / "best_checkpoints_index.csv"),
            "best_checkpoints_missing": str(all_metrics_dir / "best_checkpoints_missing.csv"),
        },
    }
    (all_metrics_dir / "best_checkpoints_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"[OK] IDs and checkpoint collection complete in: {all_metrics_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

