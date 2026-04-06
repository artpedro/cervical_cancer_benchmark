from __future__ import annotations

import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from datasets.datasets import NORM_STATS_PATH, scan_herlev, scan_riva, scan_sipakmed
from training.io_utils import env_path, setup_run_dir, tee_log
from training.pipeline import scan_dataset, train_mixed_dataset_v2

# =============================================================================
# CONFIG — edit here
# =============================================================================

SEED = 42
NUM_FOLDS = 5
BALANCE_MODE = "weighted_loss"

BATCH_SIZE = 32
NUM_WORKERS = 12

METRICS_DIR = env_path("METRICS_DIR", "workspace", "metricsv2")
RUNS_DIR = env_path("RUNS_DIR", "workspace", "runsv2")
DATA_ROOT = env_path("DATA_ROOT", "datasets", "data")

RESULTS_CSV_NAME = "training_time_results_mixed.csv"
PRINT_EVERY_EPOCH = 1
USE_AMP = True

# All three scanners use the same CV seed / fold count so fold indices align across sources.
SCANNERS: dict[str, tuple[Path, Callable[..., pd.DataFrame]]] = {
    "sipakmed": (DATA_ROOT / "sipakmed", scan_sipakmed),
    "riva": (DATA_ROOT / "riva", scan_riva),
    "herlev": (DATA_ROOT / "herlev", scan_herlev),
}

# Pairwise mixed training: (name_a, name_b) — train/val folds use both train_dev splits;
# test evaluates on the concatenated test splits from both datasets.
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
        display_name="GhostNet",
        backbone_id="ghostnet",
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
        display_name="EfficientNet-B4",
        backbone_id="efficientnet_b4",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientNet-B5",
        backbone_id="efficientnet_b5",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientNet-B6",
        backbone_id="efficientnet_b6",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
    ),
    ModelTrainConfig(
        display_name="EfficientNet-B7",
        backbone_id="efficientnet_b7",
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
        display_name="MobileNet v2 (torchvision)",
        backbone_id="tv_mobilenet_v2",
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


def _effective_mixed_pairs() -> tuple[list[tuple[str, str]], list[str]]:
    """Pairs where both dataset roots exist; plus list of missing paths."""
    missing_paths = [str(SCANNERS[n][0]) for n in SCANNERS if not _dataset_root_exists(n)]
    available = {n for n in SCANNERS if _dataset_root_exists(n)}
    effective = [(a, b) for a, b in MIXED_PAIRS if a in available and b in available]
    return effective, missing_paths


def main() -> None:
    mixed_run, missing_paths = _effective_mixed_pairs()
    if not mixed_run:
        raise FileNotFoundError(
            "No mixed pair can run: every dataset root is missing. Set DATA_ROOT or add:\n"
            + "\n".join(f"  - {p}" for p in missing_paths)
        )

    print("\n" + "=" * 70)
    print(
        "train_mixed_models — mixed two-source training, aligned CV folds, "
        "combined val + test (per-source test F1 in summary)"
    )
    print("=" * 70)
    print(f"BALANCE_MODE: {BALANCE_MODE}")
    print(f"Configured mixed pairs: {MIXED_PAIRS}")
    print(f"Data root: {DATA_ROOT.resolve()}")
    if missing_paths:
        print("\n[WARN] Missing dataset root(s); skipping pairs that need them:")
        for p in missing_paths:
            print(f"       - {p}")
    print(f"\nMixed runs (this session): {mixed_run}")
    print("=" * 70 + "\n")

    run_start_dt = datetime.datetime.now()
    run_start_str = run_start_dt.strftime("%Y-%m-%d %H:%M:%S")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RUNS_DIR / f"terminal_mixed_{run_start_dt.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    results_csv = METRICS_DIR / RESULTS_CSV_NAME

    total_configs = sum(
        len(merge_model_configs_for_pair(a, b)) * NUM_FOLDS for a, b in mixed_run
    )
    done_configs = [0]

    def progress_cb(**info):
        done_configs[0] += 1
        print(f"[PROGRESS] {done_configs[0]}/{total_configs} configs | start={run_start_str}")

    log_f, old_stdout, old_stderr = tee_log(log_path)
    try:
        print(f"[START] {run_start_str}")
        print(f"[LOG]   {log_path}")
        print(f"[CSV]   {results_csv}")

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
        print("train_mixed_models COMPLETE (all listed pairs)")
        print("=" * 70 + "\n")
    finally:
        import sys

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()


if __name__ == "__main__":
    main()
