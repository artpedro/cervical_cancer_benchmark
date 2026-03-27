from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def _discover_metrics_dirs(workspace_dir: Path) -> list[Path]:
    return sorted(
        p for p in workspace_dir.iterdir() if p.is_dir() and p.name.startswith("metrics")
    )


def _classify_csv(csv_path: Path) -> str:
    name = csv_path.name
    if name == "epoch_metrics.csv" and any(
        part.startswith("fold_") for part in csv_path.parts
    ):
        return "fold_epoch_metrics"
    if name.startswith("summary_") and name.endswith("_weighted_loss.csv"):
        return "summary_weighted_loss"
    if name.startswith("epoch_logs_") and name.endswith("_weighted_loss.csv"):
        return "epoch_logs_weighted_loss"
    if name.startswith("training_time_results") and name.endswith(".csv"):
        return "training_time_results"
    return f"other_{name.replace('.csv', '')}"


def _extract_path_metadata(rel_csv_path: Path, metrics_root_name: str) -> dict[str, str]:
    rel_parts = list(rel_csv_path.parts)
    dataset = ""
    balance_mode = ""
    run_timestamp = ""

    if len(rel_parts) >= 4 and rel_parts[1] == "weighted_loss":
        dataset = rel_parts[0]
        balance_mode = rel_parts[1]
        run_timestamp = rel_parts[2]

    fold = ""
    for part in rel_parts:
        if re.fullmatch(r"fold_\d+", part):
            fold = part
            break

    return {
        "source_metrics_root": metrics_root_name,
        "source_csv_relpath": str(rel_csv_path),
        "source_dataset_path": dataset,
        "source_balance_mode_path": balance_mode,
        "source_run_timestamp_path": run_timestamp,
        "source_fold_path": fold,
    }


@dataclass(frozen=True)
class GroupKey:
    kind: str
    columns_sig: tuple[str, ...]


def _iter_csvs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.csv")):
        if p.is_file():
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy all workspace metrics* folders and build merged all_metrics CSVs "
            "from the copy with schema-safe concatenation."
        )
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("workspace"),
        help="Workspace directory that contains metrics* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination folder for copied metrics and merged outputs. "
            "Default: workspace/all_metrics_copy_<timestamp>"
        ),
    )
    args = parser.parse_args()

    workspace_dir = args.workspace_dir.resolve()
    if not workspace_dir.exists():
        raise FileNotFoundError(f"Workspace directory does not exist: {workspace_dir}")

    metrics_dirs = _discover_metrics_dirs(workspace_dir)
    if not metrics_dirs:
        raise RuntimeError(f"No folders starting with 'metrics' found in: {workspace_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (workspace_dir / f"all_metrics_copy_{timestamp}").resolve()
    )
    copied_root = out_dir / "copied_metrics"
    merged_root = out_dir / "all_metrics"
    copied_root.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    copied_metrics_dirs: list[Path] = []
    print(f"[1/3] Copying metrics folders into: {copied_root}")
    for src in metrics_dirs:
        dst = copied_root / src.name
        shutil.copytree(src, dst, dirs_exist_ok=False)
        copied_metrics_dirs.append(dst)
        print(f"  - copied {src.name}")

    print("[2/3] Reading and grouping CSV files (schema-safe)")
    grouped_frames: dict[GroupKey, list[pd.DataFrame]] = defaultdict(list)
    kind_to_schemas: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    source_csv_count = 0
    row_count_total = 0

    for metrics_root in copied_metrics_dirs:
        for csv_path in _iter_csvs(metrics_root):
            rel_csv_path = csv_path.relative_to(metrics_root)
            kind = _classify_csv(rel_csv_path)

            df = pd.read_csv(csv_path, low_memory=False)
            cols = tuple(str(c) for c in df.columns)
            key = GroupKey(kind=kind, columns_sig=cols)

            meta = _extract_path_metadata(rel_csv_path, metrics_root.name)
            for k, v in reversed(list(meta.items())):
                df.insert(0, k, v)

            grouped_frames[key].append(df)
            kind_to_schemas[kind].add(cols)
            source_csv_count += 1
            row_count_total += len(df)

    print("[3/3] Writing merged all_metrics CSV outputs")
    written_files: list[dict[str, object]] = []
    by_kind: dict[str, list[GroupKey]] = defaultdict(list)
    for key in grouped_frames:
        by_kind[key.kind].append(key)

    for kind in sorted(by_kind):
        keys = sorted(by_kind[kind], key=lambda k: (len(k.columns_sig), k.columns_sig))
        multi_schema = len(keys) > 1
        for idx, key in enumerate(keys, start=1):
            out_name = (
                f"all_{kind}.csv"
                if not multi_schema
                else f"all_{kind}__schema_{idx:02d}.csv"
            )
            out_path = merged_root / out_name
            merged_df = pd.concat(grouped_frames[key], axis=0, ignore_index=True)
            merged_df.to_csv(out_path, index=False)

            written_files.append(
                {
                    "path": str(out_path),
                    "kind": kind,
                    "schema_index": idx if multi_schema else 1,
                    "schema_columns": list(key.columns_sig),
                    "rows": int(len(merged_df)),
                    "source_file_count": int(len(grouped_frames[key])),
                }
            )
            print(f"  - wrote {out_name} | rows={len(merged_df)} | files={len(grouped_frames[key])}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace_dir": str(workspace_dir),
        "output_dir": str(out_dir),
        "copied_metrics_root": str(copied_root),
        "merged_root": str(merged_root),
        "copied_metrics_folders": [p.name for p in copied_metrics_dirs],
        "source_csv_count": source_csv_count,
        "source_row_count_total": row_count_total,
        "kinds_detected": sorted(kind_to_schemas.keys()),
        "files_written": written_files,
    }
    manifest_path = merged_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  - wrote manifest.json")

    print("\nDone.")
    print(f"Copy root:   {copied_root}")
    print(f"All metrics: {merged_root}")


if __name__ == "__main__":
    main()

