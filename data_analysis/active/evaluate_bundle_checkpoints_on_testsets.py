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


METRIC_KEYS = ["loss", "acc", "prec", "rec", "spec", "f1", "ppv", "npv", "seconds"]

# ============================================================
# CONFIG (constants-driven run; no CLI required)
# ============================================================
ANALYSIS_DIR = Path("workspace/analysis")
WORKSPACE_BUNDLE_DIR = ANALYSIS_DIR / "all_metrics_dedup_bundle"
DATA_ROOT = Path("datasets/data")
STATS_PATH = NORM_STATS_PATH

SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
NUM_WORKERS = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set to None to evaluate all checkpoints.
LIMIT: int | None = None


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


def _scan_dataset(
    dataset: str,
    root: Path,
    *,
    num_folds: int,
    seed: int,
    test_size: float,
) -> pd.DataFrame:
    if dataset == "herlev":
        return scan_herlev(root=root, num_folds=num_folds, seed=seed, test_size=test_size)
    if dataset == "sipakmed":
        return scan_sipakmed(root=root, num_folds=num_folds, seed=seed, test_size=test_size)
    if dataset == "riva":
        return scan_riva(root=root, num_folds=num_folds, seed=seed, test_size=test_size)
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
            return ckpt  # likely raw state dict
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _build_model_for_checkpoint(
    *,
    origin: str,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    source, canonical = _canonical_from_origin(origin)
    state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

    # EAT may have different img_size configs across runs. Try common values.
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

    # For all other models, origin canonical name maps to load_any aliases/registry keys.
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


def main() -> None:
    bundle_dir = WORKSPACE_BUNDLE_DIR.resolve()
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
    merged_all = summary_df.merge(
        index_df[["cfg_id", "dataset", "model", "origin", "fold", "best_epoch"]],
        on=["dataset", "model", "origin", "fold", "best_epoch"],
        how="inner",
    )
    if merged_all.empty:
        raise RuntimeError("No configurations available after joining summary and checkpoint index.")

    merged_all["fold"] = pd.to_numeric(merged_all["fold"], errors="coerce")
    merged_all["best_epoch"] = pd.to_numeric(merged_all["best_epoch"], errors="coerce")
    merged_all = merged_all.dropna(subset=["fold", "best_epoch"]).copy()
    merged_all["fold"] = merged_all["fold"].astype(int)
    merged_all["best_epoch"] = merged_all["best_epoch"].astype(int)
    merged_all = merged_all.sort_values(["dataset", "model", "fold"]).reset_index(drop=True)

    # Infer folds from full merged table, even when evaluating a limited subset.
    dataset_num_folds = (
        merged_all.groupby("dataset")["fold"].max().astype(int).add(1).to_dict()
    )
    dataset_num_folds = {
        ds: max(2, n_folds) for ds, n_folds in dataset_num_folds.items()
    }

    merged = merged_all.copy()

    if LIMIT is not None:
        merged = merged.head(LIMIT).copy()

    device = torch.device(DEVICE)
    criterion = nn.CrossEntropyLoss()
    pin_memory = device.type == "cuda"
    use_amp = device.type == "cuda"

    # Scan each dataset once with num_folds inferred from selected rows.
    dataset_df: dict[str, pd.DataFrame] = {}
    for dataset in sorted(merged["dataset"].unique()):
        inferred_num_folds = int(dataset_num_folds[dataset])
        root = _resolve_dataset_root(DATA_ROOT.resolve(), dataset)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found for {dataset}: {root}")
        print(
            f"[scan] dataset={dataset} root={root} seed={SEED} "
            f"test_size={TEST_SIZE} num_folds={inferred_num_folds}"
        )
        dataset_df[dataset] = _scan_dataset(
            dataset,
            root,
            num_folds=inferred_num_folds,
            seed=SEED,
            test_size=TEST_SIZE,
        )

    out_dir = ANALYSIS_DIR.resolve() / "test_eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(merged)
    for i, row in merged.iterrows():
        cfg_id = str(row["cfg_id"])
        dataset = str(row["dataset"])
        model = str(row["model"])
        origin = str(row["origin"])
        fold = int(row["fold"])
        best_epoch = int(row["best_epoch"])

        ckpt_path = _checkpoint_path_for_cfg(bundle_dir, cfg_id)
        print(
            f"[{i+1}/{total}] eval cfg={cfg_id} dataset={dataset} model={model} "
            f"fold={fold} epoch={best_epoch}"
        )

        _, eval_tf = make_tf_from_stats_for_fold(dataset, fold, STATS_PATH.resolve())

        df = dataset_df[dataset]
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        if test_df.empty:
            raise RuntimeError(f"Empty test split for dataset={dataset}")

        loader = DataLoader(
            PapDataset(test_df, eval_tf),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

        model_obj = _build_model_for_checkpoint(
            origin=origin,
            checkpoint_path=ckpt_path,
            device=device,
        )
        metrics = run_epoch(
            dataloader=loader,
            model=model_obj,
            criterion=criterion,
            split_name="test",
            optimiser=None,
            scaler=None,
            use_amp=use_amp,
            device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out = {
            "cfg_id": cfg_id,
            "dataset": dataset,
            "model": model,
            "origin": origin,
            "fold": fold,
            "best_epoch": best_epoch,
            "checkpoint_path": str(ckpt_path),
            "n_test_samples": int(len(test_df)),
        }
        for k in METRIC_KEYS:
            out[f"test_{k}"] = float(metrics[k])
        rows.append(out)

    per_ckpt = pd.DataFrame(rows)
    per_ckpt_path = out_dir / "per_checkpoint_test_metrics.csv"
    per_ckpt.to_csv(per_ckpt_path, index=False)

    grouped = (
        per_ckpt.groupby(["dataset", "model", "origin"], as_index=False)[
            [f"test_{k}" for k in METRIC_KEYS]
        ]
        .agg(["mean", "std"])
    )
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    grouped_path = out_dir / "aggregated_by_model_test_metrics.csv"
    grouped.to_csv(grouped_path, index=False)

    print(f"[OK] Wrote per-checkpoint metrics: {per_ckpt_path}")
    print(f"[OK] Wrote aggregated metrics:   {grouped_path}")


if __name__ == "__main__":
    main()

