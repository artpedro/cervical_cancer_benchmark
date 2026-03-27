from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ptflops import get_model_complexity_info
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


# ============================================================
# CONFIG
# ============================================================
ANALYSIS_DIR = Path("workspace/analysis")
BUNDLE_DIR = ANALYSIS_DIR / "all_metrics_dedup_bundle"
DATA_ROOT = Path("datasets/data")
STATS_PATH = NORM_STATS_PATH

SEED = 42
TEST_SIZE = 0.2
NUM_FOLDS = 5

LATENCY_BATCH_SIZE = 16
LATENCY_NUM_BATCHES = 50
LATENCY_MEAN_LAST_K = 5
NUM_WORKERS = 0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set None to process all checkpoints.
LIMIT_CHECKPOINTS: int | None = None

OUT_DIR = ANALYSIS_DIR / "efficiency_profile"
OUT_CSV_PER_CHECKPOINT = OUT_DIR / "per_checkpoint_efficiency.csv"
OUT_CSV_AGG_MODEL = OUT_DIR / "aggregated_efficiency_by_model.csv"


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
) -> torch.nn.Module:
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


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _compute_macs_flops(model: torch.nn.Module) -> tuple[float, float]:
    # ptflops expects model on CPU for most stable behavior.
    model_cpu = model.to("cpu").eval()
    macs, _params = get_model_complexity_info(
        model_cpu,
        (3, 224, 224),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    # FLOPs convention commonly used in papers: 1 MAC ~= 2 FLOPs
    flops = 2.0 * float(macs)
    return float(macs), flops


def _measure_latency_ms(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    n_batches: int,
    mean_last_k: int,
) -> tuple[float, list[float], float, float, list[float]]:
    if mean_last_k > n_batches:
        raise ValueError("mean_last_k must be <= n_batches")

    model.eval()
    times_ms: list[float] = []
    mem_mb: list[float] = []
    it = iter(loader)

    with torch.inference_mode():
        for _ in range(n_batches):
            try:
                x, _y = next(it)
            except StopIteration:
                it = iter(loader)
                x, _y = next(it)

            x = x.to(device, non_blocking=(device.type == "cuda"))

            if device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_peak_memory_stats(device=device)
                starter.record()
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(float(starter.elapsed_time(ender)))
                mem_mb.append(float(torch.cuda.max_memory_allocated(device=device) / (1024.0**2)))
            else:
                t0 = time.perf_counter()
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
                mem_mb.append(float("nan"))

    tail = times_ms[-mean_last_k:]
    mem_tail = mem_mb[-mean_last_k:]
    mean_tail = float(sum(tail) / len(tail))
    valid_mem_tail = [m for m in mem_tail if m == m]
    valid_mem_all = [m for m in mem_mb if m == m]
    mean_mem_tail = (
        float(sum(valid_mem_tail) / len(valid_mem_tail))
        if valid_mem_tail
        else float("nan")
    )
    max_mem_all = max(valid_mem_all) if valid_mem_all else float("nan")
    return mean_tail, times_ms, mean_mem_tail, max_mem_all, mem_mb


def main() -> None:
    if LATENCY_MEAN_LAST_K > LATENCY_NUM_BATCHES:
        raise ValueError("LATENCY_MEAN_LAST_K cannot be greater than LATENCY_NUM_BATCHES")

    bundle_dir = BUNDLE_DIR.resolve()
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

    ck_df = summary_df.merge(
        index_df[["cfg_id", "dataset", "model", "origin", "fold", "best_epoch", "status"]],
        on=["dataset", "model", "origin", "fold", "best_epoch"],
        how="inner",
    )
    ck_df = ck_df[ck_df["status"] == "copied"].copy()
    if ck_df.empty:
        raise RuntimeError("No copied checkpoints found after join.")

    ck_df["fold"] = pd.to_numeric(ck_df["fold"], errors="coerce")
    ck_df["best_epoch"] = pd.to_numeric(ck_df["best_epoch"], errors="coerce")
    ck_df = ck_df.dropna(subset=["fold", "best_epoch"]).copy()
    ck_df["fold"] = ck_df["fold"].astype(int)
    ck_df["best_epoch"] = ck_df["best_epoch"].astype(int)
    ck_df = ck_df.sort_values(["dataset", "model", "fold"]).reset_index(drop=True)

    if LIMIT_CHECKPOINTS is not None:
        ck_df = ck_df.head(LIMIT_CHECKPOINTS).copy()

    device = torch.device(DEVICE)

    # Scan each dataset once.
    dataset_frames: dict[str, pd.DataFrame] = {}
    for ds in sorted(ck_df["dataset"].astype(str).unique().tolist()):
        root = _resolve_dataset_root(DATA_ROOT.resolve(), ds)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found for {ds}: {root}")
        print(f"[scan] dataset={ds} root={root}")
        dataset_frames[ds] = _scan_dataset(ds, root)

    rows: list[dict[str, Any]] = []
    total = len(ck_df)
    for i, row in ck_df.iterrows():
        cfg_id = str(row["cfg_id"])
        dataset = str(row["dataset"])
        model_name = str(row["model"])
        origin = str(row["origin"])
        fold = int(row["fold"])
        best_epoch = int(row["best_epoch"])

        ckpt_path = _checkpoint_path_for_cfg(bundle_dir, cfg_id)
        print(
            f"[{i+1}/{total}] profiling cfg={cfg_id} dataset={dataset} model={model_name} fold={fold}"
        )

        _, eval_tf = make_tf_from_stats_for_fold(dataset, fold, STATS_PATH.resolve())
        df_test = dataset_frames[dataset]
        df_test = df_test[df_test["split"] == "test"].reset_index(drop=True)
        if df_test.empty:
            raise RuntimeError(f"Empty test split for dataset={dataset}")

        loader = DataLoader(
            PapDataset(df_test, eval_tf),
            batch_size=LATENCY_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        model = _build_model_for_checkpoint(
            origin=origin,
            checkpoint_path=ckpt_path,
            device=device,
        )

        n_params = _count_params(model)
        macs, flops = _compute_macs_flops(model)

        # Put model back on target device after ptflops CPU pass.
        model.to(device).eval()
        (
            latency_mean_ms_last5,
            all_times_ms,
            memory_mean_mb_last5,
            memory_peak_mb_max10,
            all_mem_mb,
        ) = _measure_latency_ms(
            model=model,
            loader=loader,
            device=device,
            n_batches=LATENCY_NUM_BATCHES,
            mean_last_k=LATENCY_MEAN_LAST_K,
        )

        rows.append(
            {
                "cfg_id": cfg_id,
                "dataset": dataset,
                "model": model_name,
                "origin": origin,
                "fold": fold,
                "best_epoch": best_epoch,
                "checkpoint_path": str(ckpt_path),
                "n_test_samples": int(len(df_test)),
                "params_count": int(n_params),
                "macs_count": float(macs),
                "flops_count": float(flops),
                "latency_batch_size": LATENCY_BATCH_SIZE,
                "latency_num_batches": LATENCY_NUM_BATCHES,
                "latency_mean_last_k": LATENCY_MEAN_LAST_K,
                "latency_mean_ms_last_k": float(latency_mean_ms_last5),
                "latency_all_10_batches_ms": ";".join(f"{t:.6f}" for t in all_times_ms),
                "memory_mean_mb_last_k": float(memory_mean_mb_last5),
                "memory_peak_mb_max_10_batches": float(memory_peak_mb_max10),
                "memory_all_10_batches_mb": ";".join(
                    "nan" if (m != m) else f"{m:.6f}" for m in all_mem_mb
                ),
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ckpt = pd.DataFrame(rows)
    per_ckpt.to_csv(OUT_CSV_PER_CHECKPOINT.resolve(), index=False)

    agg = (
        per_ckpt.groupby(["dataset", "model", "origin"], as_index=False)[
            [
                "params_count",
                "macs_count",
                "flops_count",
                "latency_mean_ms_last_k",
                "memory_mean_mb_last_k",
                "memory_peak_mb_max_10_batches",
            ]
        ]
        .agg(["mean", "std"])
    )
    agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in agg.columns
    ]
    agg.to_csv(OUT_CSV_AGG_MODEL.resolve(), index=False)

    print(f"[OK] wrote {OUT_CSV_PER_CHECKPOINT.resolve()} ({len(per_ckpt)} rows)")
    print(f"[OK] wrote {OUT_CSV_AGG_MODEL.resolve()} ({len(agg)} rows)")


if __name__ == "__main__":
    main()

