from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.datasets import (
    NORM_STATS_PATH,
    PapDataset,
    make_tf_from_stats_for_fold,
    scan_herlev,
    scan_riva,
    scan_sipakmed,
)
from model_loader import load_any
from training.engine import run_epoch


# ============================================================
# CONFIG
# ============================================================
ANALYSIS_DIR = Path("workspace/analysis")
BUNDLE_DIR = ANALYSIS_DIR / "all_metrics_dedup_bundle"
DATA_ROOT = Path("datasets/data")
STATS_PATH = NORM_STATS_PATH

ALL_DATASETS = ("herlev", "sipakmed", "riva")
FULL_TARGET_DATASETS = {"riva", "herlev","sipakmed"}  # extra "entire dataset" evaluation
INCLUDE_SAME_DATASET = False  # cross-dataset only

SEED = 42
TEST_SIZE = 0.2
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 12
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Optional quick smoke run; set to None to evaluate all checkpoints.
LIMIT_CHECKPOINTS: int | None = None

OUT_DIR = ANALYSIS_DIR / "test_eval_results_cross_dataset"
OUT_CSV_TEST_SPLIT = OUT_DIR / "cross_dataset_testsplit_metrics.csv"
OUT_CSV_FULL_RIVA_HERLEV = OUT_DIR / "cross_dataset_full_riva_herlev_metrics.csv"

METRIC_KEYS = ["loss", "acc", "prec", "rec", "spec", "f1", "ppv", "npv", "seconds"]


def _resolve_dataset_root(data_root: Path, dataset: str) -> Path:
    if dataset == "herlev":
        return data_root / "smear2005"
    if dataset == "sipakmed":
        return data_root / "sipakmed"
    if dataset == "riva":
        low = data_root / "riva"
        up = data_root / "RIVA"
        return low if low.exists() else up
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _scan_dataset(dataset: str, root: Path) -> pd.DataFrame:
    if dataset == "herlev":
        return scan_herlev(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    if dataset == "sipakmed":
        return scan_sipakmed(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    if dataset == "riva":
        return scan_riva(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _canonical_from_origin(origin: str) -> tuple[str, str]:
    if ":" in origin:
        source, canonical = origin.split(":", 1)
        return source, canonical
    return "", origin


def _load_state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _build_model_for_checkpoint(
    *,
    origin: str,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    source, canonical = _canonical_from_origin(origin)
    state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

    if source == "custom" and canonical == "eat":
        last_err: Exception | None = None
        for img_size in (224, 144, 192, 256):
            model, _, _, _ = load_any(
                "eat",
                num_classes=2,
                pretrained=False,
                device=device,
                checkpoint_path=None,
                img_size=img_size,
            )
            try:
                model.load_state_dict(state_dict, strict=True)
                model.to(device)
                model.eval()
                return model
            except Exception as e:  # pylint: disable=broad-except
                last_err = e
        raise RuntimeError(
            f"Failed loading EAT checkpoint {checkpoint_path} with tested img sizes. Last error: {last_err}"
        )

    model, _, _, _ = load_any(
        canonical,
        num_classes=2,
        pretrained=False,
        device=device,
        checkpoint_path=None,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _checkpoint_path_for_cfg(bundle_dir: Path, cfg_id: str) -> Path:
    ckpt_dir = bundle_dir / "best_checkpoints"
    matches = sorted(ckpt_dir.glob(f"{cfg_id}__*.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint file found in {ckpt_dir} for cfg_id={cfg_id}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple checkpoint files found for cfg_id={cfg_id}: {[str(m) for m in matches]}"
        )
    return matches[0]


def _evaluate_subset(
    *,
    model: nn.Module,
    eval_tf,
    df_subset: pd.DataFrame,
    device: torch.device,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        PapDataset(df_subset.reset_index(drop=True), eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    metrics = run_epoch(
        dataloader=loader,
        model=model,
        criterion=criterion,
        split_name="test",
        optimiser=None,
        scaler=None,
        use_amp=(device.type == "cuda"),
        device=device,
    )
    return metrics


def main() -> None:
    bundle_dir = BUNDLE_DIR.resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    summary_path = bundle_dir / "deduplicated_best" / "all_summary_weighted_loss.dedup_best.csv"
    index_path = bundle_dir / "best_checkpoints_index.csv"
    if not summary_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            "Missing required inputs:\n"
            f"- {summary_path}\n"
            f"- {index_path}"
        )

    summary_df = pd.read_csv(summary_path, low_memory=False)
    index_df = pd.read_csv(index_path, low_memory=False)

    checkpoints_df = summary_df.merge(
        index_df[["cfg_id", "dataset", "model", "origin", "fold", "best_epoch", "status"]],
        on=["dataset", "model", "origin", "fold", "best_epoch"],
        how="inner",
    )
    checkpoints_df = checkpoints_df[checkpoints_df["status"] == "copied"].copy()
    if checkpoints_df.empty:
        raise RuntimeError("No copied checkpoints available after merge.")

    checkpoints_df["fold"] = pd.to_numeric(checkpoints_df["fold"], errors="coerce")
    checkpoints_df["best_epoch"] = pd.to_numeric(checkpoints_df["best_epoch"], errors="coerce")
    checkpoints_df = checkpoints_df.dropna(subset=["fold", "best_epoch"]).copy()
    checkpoints_df["fold"] = checkpoints_df["fold"].astype(int)
    checkpoints_df["best_epoch"] = checkpoints_df["best_epoch"].astype(int)
    checkpoints_df = checkpoints_df.sort_values(["dataset", "model", "fold"]).reset_index(drop=True)

    if LIMIT_CHECKPOINTS is not None:
        checkpoints_df = checkpoints_df.head(LIMIT_CHECKPOINTS).copy()

    # Scan all target datasets once.
    dataset_frames: dict[str, pd.DataFrame] = {}
    data_root = DATA_ROOT.resolve()
    for ds in ALL_DATASETS:
        root = _resolve_dataset_root(data_root, ds)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found for {ds}: {root}")
        print(f"[scan] dataset={ds} root={root}")
        dataset_frames[ds] = _scan_dataset(ds, root)

    device = torch.device(DEVICE)
    rows_testsplit: list[dict[str, Any]] = []
    rows_full: list[dict[str, Any]] = []

    total = len(checkpoints_df)
    for i, ck in checkpoints_df.iterrows():
        cfg_id = str(ck["cfg_id"])
        source_dataset = str(ck["dataset"])
        source_model = str(ck["model"])
        source_origin = str(ck["origin"])
        source_fold = int(ck["fold"])
        best_epoch = int(ck["best_epoch"])

        ckpt_path = _checkpoint_path_for_cfg(bundle_dir, cfg_id)
        _, eval_tf = make_tf_from_stats_for_fold(source_dataset, source_fold, STATS_PATH.resolve())

        model_obj = _build_model_for_checkpoint(
            origin=source_origin,
            checkpoint_path=ckpt_path,
            device=device,
        )

        target_datasets = list(ALL_DATASETS)
        if not INCLUDE_SAME_DATASET:
            target_datasets = [d for d in target_datasets if d != source_dataset]

        print(
            f"[{i+1}/{total}] source={source_dataset}/{source_model}/fold{source_fold} "
            f"targets={target_datasets}"
        )

        for target_dataset in target_datasets:
            df_target = dataset_frames[target_dataset]

            # 1) cross-dataset on target test split
            df_test = df_target[df_target["split"] == "test"].reset_index(drop=True)
            if df_test.empty:
                raise RuntimeError(f"Empty test split for target dataset={target_dataset}")
            m_test = _evaluate_subset(
                model=model_obj,
                eval_tf=eval_tf,
                df_subset=df_test,
                device=device,
            )
            rec_test = {
                "cfg_id": cfg_id,
                "source_dataset": source_dataset,
                "source_model": source_model,
                "source_origin": source_origin,
                "source_fold": source_fold,
                "best_epoch": best_epoch,
                "checkpoint_path": str(ckpt_path),
                "target_dataset": target_dataset,
                "target_scope": "test_split",
                "n_target_samples": int(len(df_test)),
            }
            for k in METRIC_KEYS:
                rec_test[f"target_{k}"] = float(m_test[k])
            rows_testsplit.append(rec_test)

            # 2) extra full-dataset evaluation on riva/herlev targets
            if target_dataset in FULL_TARGET_DATASETS:
                df_full = df_target.reset_index(drop=True)
                if df_full.empty:
                    raise RuntimeError(f"Empty full dataset for target dataset={target_dataset}")
                m_full = _evaluate_subset(
                    model=model_obj,
                    eval_tf=eval_tf,
                    df_subset=df_full,
                    device=device,
                )
                rec_full = {
                    "cfg_id": cfg_id,
                    "source_dataset": source_dataset,
                    "source_model": source_model,
                    "source_origin": source_origin,
                    "source_fold": source_fold,
                    "best_epoch": best_epoch,
                    "checkpoint_path": str(ckpt_path),
                    "target_dataset": target_dataset,
                    "target_scope": "full_dataset",
                    "n_target_samples": int(len(df_full)),
                }
                for k in METRIC_KEYS:
                    rec_full[f"target_{k}"] = float(m_full[k])
                rows_full.append(rec_full)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_testsplit = pd.DataFrame(rows_testsplit)
    df_full = pd.DataFrame(rows_full)

    df_testsplit.to_csv(OUT_CSV_TEST_SPLIT.resolve(), index=False)
    df_full.to_csv(OUT_CSV_FULL_RIVA_HERLEV.resolve(), index=False)

    print(f"[OK] wrote {OUT_CSV_TEST_SPLIT.resolve()} ({len(df_testsplit)} rows)")
    print(f"[OK] wrote {OUT_CSV_FULL_RIVA_HERLEV.resolve()} ({len(df_full)} rows)")


if __name__ == "__main__":
    main()

