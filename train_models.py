from __future__ import annotations

import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from datasets.datasets import NORM_STATS_PATH, scan_riva, scan_sipakmed
from training.io_utils import env_path, setup_run_dir, tee_log
from training.pipeline import scan_dataset, train_dataset_v2

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

RESULTS_CSV_NAME = "training_time_results_v2.csv"
PRINT_EVERY_EPOCH = 1
USE_AMP = True

DATASETS: list[tuple[str, Path, Callable]] = [
    ("sipakmed", DATA_ROOT / "sipakmed", scan_sipakmed),
    ("riva", DATA_ROOT / "riva", scan_riva),
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


SIPAKMED_MODEL_CONFIGS: list[ModelTrainConfig] = [
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
]

RIVA_MODEL_CONFIGS: list[ModelTrainConfig] = [
    ModelTrainConfig(
        display_name="EfficientFormerV2 S1",
        backbone_id="efficientformerv2_s1",
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
        display_name="EAT",
        backbone_id="eat",
        epochs=100,
        lr=1e-3,
        scheduler_milestones=[25, 50, 75],
        scheduler_gamma=0.5,
        load_kwargs={"img_size": 224},
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
]

MODEL_CONFIGS_BY_DATASET: dict[str, list[ModelTrainConfig]] = {
    "sipakmed": SIPAKMED_MODEL_CONFIGS,
    "riva": RIVA_MODEL_CONFIGS,
}

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


def main() -> None:
    print("\n" + "=" * 70)
    print("train_models_v2 — weighted_loss only, 5-fold CV, per-model epochs & schedule")
    print("=" * 70)
    print(f"BALANCE_MODE: {BALANCE_MODE}")
    print(f"Datasets: {[d[0] for d in DATASETS]}")
    print(f"Data root: {DATA_ROOT.resolve()}")
    for dataset_name, _, _ in DATASETS:
        model_names = [c.display_name for c in MODEL_CONFIGS_BY_DATASET[dataset_name]]
        print(f"Models for {dataset_name}: {model_names}")
    print("=" * 70 + "\n")

    missing_roots = [str(root) for _, root, _ in DATASETS if not root.exists()]
    if missing_roots:
        raise FileNotFoundError(
            "Missing dataset roots. Set DATA_ROOT or create these paths:\n"
            + "\n".join(f"  - {p}" for p in missing_roots)
        )

    run_start_dt = datetime.datetime.now()
    run_start_str = run_start_dt.strftime("%Y-%m-%d %H:%M:%S")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RUNS_DIR / f"terminal_v2_{run_start_dt.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    results_csv = METRICS_DIR / RESULTS_CSV_NAME

    total_configs = sum(
        len(MODEL_CONFIGS_BY_DATASET[name]) * NUM_FOLDS for name, _, _ in DATASETS
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

        for name, root, scanner in DATASETS:
            model_configs = MODEL_CONFIGS_BY_DATASET[name]
            print(f"\nScanning dataset: {name} at {root}")
            df = scan_dataset(name, root, scanner, num_folds=NUM_FOLDS, seed=SEED)

            print("\n" + "-" * 70)
            print(f"Starting training on {name} | {BALANCE_MODE}")
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

        print("\n" + "=" * 70)
        print("train_models_v2 COMPLETE (weighted_loss, all listed datasets)")
        print("=" * 70 + "\n")
    finally:
        import sys

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()


if __name__ == "__main__":
    main()
