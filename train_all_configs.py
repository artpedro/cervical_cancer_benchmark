from __future__ import annotations

import datetime
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from datasets.datasets import NORM_STATS_PATH, scan_herlev, scan_riva, scan_sipakmed
from training.io_utils import env_path, setup_run_dir, tee_log
from training.pipeline import scan_dataset, train_dataset_v2, train_mixed_dataset_v2

# =============================================================================
# CONFIG — edit here (solo + mixed)
# =============================================================================

SEED = 42
NUM_FOLDS = 5
BALANCE_MODE = "weighted_loss"

BATCH_SIZE = 64
NUM_WORKERS = 6

METRICS_DIR = env_path("METRICS_DIR", "workspace", "metricsv2")
RUNS_DIR = env_path("RUNS_DIR", "workspace", "runsv2")
DATA_ROOT = env_path("DATA_ROOT", "datasets", "data")

RESULTS_CSV_NAME = "training_time_results_all.csv"
PRINT_EVERY_EPOCH = 1
USE_AMP = True

# Single-source training order (paths under DATA_ROOT / <name>). Missing roots are skipped
# with a warning; set DATA_ROOT or add symlinks if a dataset should be included.
SOLO_DATASET_NAMES: tuple[str, ...] = ("sipakmed", "riva", "herlev")

# All scanners share the same CV seed / fold count so fold indices align for mixed runs.
SCANNERS: dict[str, tuple[Path, Callable[..., pd.DataFrame]]] = {
    "sipakmed": (DATA_ROOT / "sipakmed", scan_sipakmed),
    "riva": (DATA_ROOT / "riva", scan_riva),
    "herlev": (DATA_ROOT / "smear2005", scan_herlev),
}

# Pairwise mixed training: (name_a, name_b).
MIXED_PAIRS: list[tuple[str, str]] = [
    ("sipakmed", "riva"),
    ("riva", "herlev"),
    ("herlev", "sipakmed"),
]


@dataclass
class ModelTrainConfig:
    display_name: str
    backbone_id: str
    epochs: int = 25
    lr: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 5e-3
    scheduler_milestones: list[int] = field(default_factory=lambda: [10, 20])
    scheduler_gamma: float = 0.1
    pretrained: bool = True
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    max_params_m: float = 100.0


# All ``model_loader.MODEL_REGISTRY`` entries (one row per canonical model), same specs.
ALL_MODEL_CONFIGS: list[ModelTrainConfig] = [
    ModelTrainConfig(
        display_name="EAT",
        backbone_id="eat",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
        load_kwargs={"img_size": 224},
    ),
    ModelTrainConfig(
        display_name="iFormer-M",
        backbone_id="iformer_m",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientFormerV2 S0",
        backbone_id="efficientformerv2_s0",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientFormerV2 S1",
        backbone_id="efficientformerv2_s1",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientFormerV2 S2",
        backbone_id="efficientformerv2_s2",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="FastViT T8",
        backbone_id="fastvit_t8",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="LeViT 128s",
        backbone_id="levit_128s",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="MobileNet v4 Conv Small",
        backbone_id="mobilenet_v4",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="MobileViT v2 100",
        backbone_id="mobilevitv2_100",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientNet-B0",
        backbone_id="efficientnet_b0",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientNet-B1",
        backbone_id="efficientnet_b1",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientNet-B2",
        backbone_id="efficientnet_b2",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientNet-B3",
        backbone_id="efficientnet_b3",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="MobileNet v2",
        backbone_id="mobilenetv2_100",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="ShuffleNet v2 1.0",
        backbone_id="tv_shufflenet_v2_x1_0",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="SqueezeNet 1.1",
        backbone_id="tv_squeezenet1_1",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
]

MODEL_CONFIGS_BY_DATASET: dict[str, list[ModelTrainConfig]] = {
    "sipakmed": ALL_MODEL_CONFIGS,
    "riva": ALL_MODEL_CONFIGS,
    "herlev": ALL_MODEL_CONFIGS,
}


def merge_model_configs_for_pair(a: str, b: str) -> list[ModelTrainConfig]:
    """Union of both datasets' model lists, deduplicated by ``backbone_id`` (first wins)."""
    seen: set[str] = set()
    out: list[ModelTrainConfig] = []
    for cfg in MODEL_CONFIGS_BY_DATASET[a] + MODEL_CONFIGS_BY_DATASET[b]:
        if cfg.backbone_id in seen:
            continue
        seen.add(cfg.backbone_id)
        out.append(cfg)
    return out


def build_mixed_dataframe(name_a: str, name_b: str) -> pd.DataFrame:
    root_a, scan_a = SCANNERS[name_a]
    root_b, scan_b = SCANNERS[name_b]
    df_a = scan_dataset(name_a, root_a, scan_a, num_folds=NUM_FOLDS, seed=SEED)
    df_b = scan_dataset(name_b, root_b, scan_b, num_folds=NUM_FOLDS, seed=SEED)
    df_a = df_a.assign(source_dataset=name_a)
    df_b = df_b.assign(source_dataset=name_b)
    return pd.concat([df_a, df_b], ignore_index=True)


def mixed_run_slug(name_a: str, name_b: str) -> str:
    return f"{name_a}_{name_b}"


# =============================================================================
# Runtime setup
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _dataset_root_exists(name: str) -> bool:
    return SCANNERS[name][0].exists()


def _effective_solo_and_mixed() -> tuple[tuple[str, ...], list[tuple[str, str]], list[str]]:
    """
    Solo names and mixed pairs that only reference datasets whose roots exist.
    Returns (solo_effective, mixed_effective, missing_paths).
    """
    missing_paths = [str(SCANNERS[n][0]) for n in SCANNERS if not _dataset_root_exists(n)]
    available = {n for n in SCANNERS if _dataset_root_exists(n)}
    solo_effective = tuple(n for n in SOLO_DATASET_NAMES if n in available)
    mixed_effective = [(a, b) for a, b in MIXED_PAIRS if a in available and b in available]
    return solo_effective, mixed_effective, missing_paths


def _total_config_runs(
    solo_names: tuple[str, ...], mixed_pairs: list[tuple[str, str]]
) -> int:
    solo = sum(
        len(MODEL_CONFIGS_BY_DATASET[name]) * NUM_FOLDS for name in solo_names
    )
    mixed = sum(
        len(merge_model_configs_for_pair(a, b)) * NUM_FOLDS for a, b in mixed_pairs
    )
    return solo + mixed


def main() -> None:
    solo_run, mixed_run, missing_paths = _effective_solo_and_mixed()
    if not solo_run and not mixed_run:
        raise FileNotFoundError(
            "No dataset roots found under DATA_ROOT. Set DATA_ROOT or create paths for "
            "at least one of:\n"
            + "\n".join(f"  - {p}" for p in missing_paths)
        )

    print("\n" + "=" * 70)
    print(
        "train_all_configs — solo (SOLO_DATASET_NAMES) then mixed pairs (MIXED_PAIRS); "
        "edit CONFIG at top of this file"
    )
    print("=" * 70)
    print(f"BALANCE_MODE: {BALANCE_MODE}")
    print(f"Data root: {DATA_ROOT.resolve()}")
    print(f"Configured solo datasets: {SOLO_DATASET_NAMES}")
    print(f"Configured mixed pairs: {MIXED_PAIRS}")
    if missing_paths:
        print("\n[WARN] Missing dataset root(s); skipping solo/mixed jobs that need them:")
        for p in missing_paths:
            print(f"       - {p}")
    print(f"\nSolo runs (this session): {solo_run or 'none'}")
    print(f"Mixed runs (this session): {mixed_run or 'none'}")
    total_configs = _total_config_runs(solo_run, mixed_run)
    print(f"Total CV config runs (progress bar): {total_configs}")
    print("=" * 70 + "\n")

    run_start_dt = datetime.datetime.now()
    run_start_str = run_start_dt.strftime("%Y-%m-%d %H:%M:%S")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RUNS_DIR / f"terminal_all_{run_start_dt.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    results_csv = METRICS_DIR / RESULTS_CSV_NAME
    done_configs = [0]

    def progress_cb(**info):
        done_configs[0] += 1
        print(f"[PROGRESS] {done_configs[0]}/{total_configs} configs | start={run_start_str}")

    log_f, old_stdout, old_stderr = tee_log(log_path)
    try:
        print(f"[START] {run_start_str}")
        print(f"[LOG]   {log_path}")
        print(f"[CSV]   {results_csv}")

        # --- Solo ---
        print("\n" + "#" * 70)
        print("# PHASE 1: single-dataset training")
        print("#" * 70 + "\n")

        if not solo_run:
            print("(No solo datasets available; skipping phase 1.)\n")

        for name in solo_run:
            root, scanner = SCANNERS[name]
            model_configs = MODEL_CONFIGS_BY_DATASET[name]
            print(f"\nScanning dataset: {name} at {root}")
            df = scan_dataset(name, root, scanner, num_folds=NUM_FOLDS, seed=SEED)

            print("\n" + "-" * 70)
            print(f"Training solo {name} | {BALANCE_MODE}")
            print("-" * 70 + "\n")

            run_dir = setup_run_dir(METRICS_DIR, dataset_name=name, balance_mode=BALANCE_MODE)

            train_dataset_v2(
                name=name,
                df=df,
                run_dir=run_dir,
                model_configs=model_configs,
                balance_mode=BALANCE_MODE,
                num_folds=NUM_FOLDS,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                stats_path=NORM_STATS_PATH,
                device=DEVICE,
                use_amp=USE_AMP,
                results_csv=results_csv,
                progress_cb=progress_cb,
                print_every_epoch=PRINT_EVERY_EPOCH,
            )

        # --- Mixed ---
        print("\n" + "#" * 70)
        print("# PHASE 2: mixed-dataset training")
        print("#" * 70 + "\n")

        if not mixed_run:
            print("(No mixed pairs available; skipping phase 2.)\n")

        for name_a, name_b in mixed_run:
            slug = mixed_run_slug(name_a, name_b)
            model_configs = merge_model_configs_for_pair(name_a, name_b)
            print(f"\nScanning mixed pair: {slug} ({name_a} + {name_b})")
            df = build_mixed_dataframe(name_a, name_b)

            print("\n" + "-" * 70)
            print(f"Training mixed {slug} | models: {[c.display_name for c in model_configs]}")
            print("-" * 70 + "\n")

            run_dir = setup_run_dir(METRICS_DIR, dataset_name=slug, balance_mode=BALANCE_MODE)

            train_mixed_dataset_v2(
                name=slug,
                df=df,
                source_names=(name_a, name_b),
                run_dir=run_dir,
                model_configs=model_configs,
                balance_mode=BALANCE_MODE,
                num_folds=NUM_FOLDS,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                stats_path=NORM_STATS_PATH,
                device=DEVICE,
                use_amp=USE_AMP,
                results_csv=results_csv,
                progress_cb=progress_cb,
                print_every_epoch=PRINT_EVERY_EPOCH,
            )

        print("\n" + "=" * 70)
        print("train_all_configs COMPLETE (solo + mixed)")
        print("=" * 70 + "\n")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()


if __name__ == "__main__":
    main()
