from __future__ import annotations

import datetime
import gc
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR

from datasets.datasets import get_loaders, make_tf_from_stats_for_fold
from model_loader import load_any
from training.engine import compute_class_weights, run_epoch
from training.eta import ModelETATracker
from training.io_utils import append_csv_rows, setup_fold_dir, write_json


def scan_dataset(name: str, root: Path, scanner: Callable, *, num_folds: int, seed: int) -> pd.DataFrame:
    if name == "apacc":
        return scanner(root=root, num_folds=num_folds, seed=seed)
    return scanner(root=root, num_folds=num_folds, seed=seed, test_size=0.2)


def train_dataset_v2(
    name: str,
    df: pd.DataFrame,
    run_dir: Path,
    model_configs: list[Any],
    *,
    balance_mode: str,
    num_folds: int,
    batch_size: int,
    num_workers: int,
    stats_path: Path,
    device: torch.device,
    use_amp: bool,
    results_csv: Path | None = None,
    progress_cb: Optional[Callable[..., None]] = None,
    print_every_epoch: int = 1,
) -> None:
    assert balance_mode == "weighted_loss"

    print(f"\n{'=' * 70}")
    print(f"DATASET: {name.upper()}  |  balance_mode = {balance_mode}  (train_models_v2)")
    print(f"Run directory: {run_dir}")
    print(f"{'=' * 70}")

    if "split" in df.columns:
        train_df = df[df["split"].isin(["train", "train_dev"])].reset_index(drop=True)
        print(
            f"Using rows with split in ('train','train_dev') for training (rows: {len(train_df)})"
        )
    else:
        train_df = df.reset_index(drop=True)
        print(f"No 'split' column found; using all {len(train_df)} rows for training.")

    class_dist = train_df["binary_idx"].value_counts().sort_index()
    total = len(train_df)
    print("\nClass Distribution (training subset):")
    print(
        f"  Class 0 (Normal):   {class_dist.get(0, 0):>6} samples ({100 * class_dist.get(0, 0) / total:5.1f}%)"
    )
    if 1 in class_dist.index:
        print(
            f"  Class 1 (Abnormal): {class_dist.get(1, 0):>6} samples ({100 * class_dist.get(1, 0) / total:5.1f}%)"
        )

    print("\n→ Standard DataLoader + CrossEntropyLoss(class_weights) per fold.")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(" GPU:", torch.cuda.get_device_name())
    amp_flag = bool(use_amp and torch.cuda.is_available() and device.type == "cuda")
    print(f"AMP enabled: {amp_flag}")

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    epoch_file = run_dir / f"epoch_logs_{name}_{balance_mode}.csv"
    summary_file = run_dir / f"summary_{name}_{balance_mode}.csv"

    epoch_cols = [
        "dataset",
        "model",
        "origin",
        "fold",
        "epoch",
        "split",
        "loss",
        "acc",
        "prec",
        "rec",
        "spec",
        "f1",
        "ppv",
        "npv",
        "lr",
        "seconds",
    ]

    summary_cols = [
        "dataset",
        "model",
        "origin",
        "fold",
        "best_epoch",
        "best_acc",
        "best_prec",
        "best_rec",
        "best_spec",
        "best_f1",
        "best_ppv",
        "best_npv",
    ]

    def _append_results(row: dict) -> None:
        if results_csv is None:
            return
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = results_csv.exists() and results_csv.stat().st_size > 0
        pd.DataFrame([row]).to_csv(results_csv, mode="a", header=not file_exists, index=False)

    for cfg in model_configs:
        friendly_name = cfg.display_name
        backbone_id = cfg.backbone_id
        eta_tracker = ModelETATracker(
            dataset=name,
            model=cfg.display_name,
            total_folds=num_folds,
            epochs_per_fold=cfg.epochs,
        )

        for fold in range(num_folds):
            fold_dir = setup_fold_dir(run_dir, friendly_name, fold)
            fold_epoch_file = fold_dir / "epoch_metrics.csv"
            fold_best_json = fold_dir / "best_metrics.json"
            fold_last_json = fold_dir / "last_epoch_metrics.json"
            best_ckpt_path = fold_dir / "checkpoints" / "best.pt"

            config_start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config_t0 = time.time()

            print(
                f"\n> {name} | {friendly_name} | fold {fold} "
                f"| model progress: "
                f"{eta_tracker.train_steps_done}/{eta_tracker.total_train_steps} train "
                f"+ {eta_tracker.val_steps_done}/{eta_tracker.total_val_steps} val "
                f"| ETA {eta_tracker.eta_hms()}"
            )

            _train_tf, _val_tf = make_tf_from_stats_for_fold(name, fold, stats_path)

            pin = torch.cuda.is_available()
            train_loader, val_loader = get_loaders(
                df=train_df,
                fold=fold,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin,
                train_tf=_train_tf,
                val_tf=_val_tf,
            )
            train_mask = train_df["fold"] != fold
            train_fold_df = train_df.loc[train_mask]
            class_weights = compute_class_weights(train_fold_df, device=device)
            print(f"  Fold {fold}: class weights = {class_weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            model, _, origin, _n_params = load_any(
                backbone_id,
                num_classes=2,
                pretrained=cfg.pretrained,
                device=device,
                max_params_m=cfg.max_params_m,
                **cfg.load_kwargs,
            )
            model.to(device)

            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
            scheduler = MultiStepLR(
                optimiser,
                milestones=list(cfg.scheduler_milestones),
                gamma=cfg.scheduler_gamma,
            )

            scaler = GradScaler("cuda", enabled=amp_flag)
            best_val = {k: 0.0 for k in ["acc", "prec", "rec", "spec", "f1", "ppv", "npv"]}
            best_val["epoch"] = 0

            for epoch in range(1, cfg.epochs + 1):
                print(f"\nEpoch {epoch:03d}/{cfg.epochs:03d}")
                lr_now = optimiser.param_groups[0]["lr"]

                train_m = run_epoch(
                    train_loader,
                    model,
                    criterion,
                    "train",
                    optimiser,
                    scaler=scaler,
                    use_amp=use_amp,
                    device=device,
                )
                val_m = run_epoch(
                    val_loader,
                    model,
                    criterion,
                    "val",
                    optimiser=None,
                    scaler=None,
                    use_amp=use_amp,
                    device=device,
                )

                eta_tracker.update_train(train_m["seconds"])
                eta_tracker.update_val(val_m["seconds"])
                model_eta = eta_tracker.eta_hms()
                duration = train_m["seconds"] + val_m["seconds"]
                scheduler.step()

                rows = []
                for split_name, m in [("train", train_m), ("val", val_m)]:
                    rows.append(
                        OrderedDict(
                            dataset=name,
                            model=friendly_name,
                            origin=origin,
                            fold=fold,
                            epoch=epoch,
                            split=split_name,
                            loss=m["loss"],
                            acc=m["acc"],
                            prec=m["prec"],
                            rec=m["rec"],
                            spec=m["spec"],
                            f1=m["f1"],
                            ppv=m["ppv"],
                            npv=m["npv"],
                            lr=lr_now,
                            seconds=m["seconds"],
                        )
                    )

                append_csv_rows(epoch_file, rows, epoch_cols)
                append_csv_rows(fold_epoch_file, rows, epoch_cols)

                write_json(
                    fold_last_json,
                    {
                        "dataset": name,
                        "model": friendly_name,
                        "origin": origin,
                        "fold": fold,
                        "epoch": epoch,
                        "lr": lr_now,
                        "seconds": duration,
                        "train": train_m,
                        "val": val_m,
                    },
                )

                if val_m["acc"] > best_val["acc"]:
                    best_val.update(val_m)
                    best_val["epoch"] = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimiser_state": optimiser.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_val": dict(best_val),
                            "dataset": name,
                            "model": friendly_name,
                            "origin": origin,
                            "fold": fold,
                        },
                        best_ckpt_path,
                    )

                    write_json(
                        fold_best_json,
                        {
                            "dataset": name,
                            "model": friendly_name,
                            "origin": origin,
                            "fold": fold,
                            "best_epoch": best_val["epoch"],
                            "best_acc": best_val["acc"],
                            "best_prec": best_val["prec"],
                            "best_rec": best_val["rec"],
                            "best_spec": best_val["spec"],
                            "best_f1": best_val["f1"],
                            "best_ppv": best_val["ppv"],
                            "best_npv": best_val["npv"],
                        },
                    )

                pe = max(1, print_every_epoch)
                if epoch == 1 or epoch % pe == 0 or epoch == cfg.epochs:
                    print(
                        f"train_loss={train_m['loss']:.4f} "
                        f"train_acc={train_m['acc']:.4f} "
                        f"train_f1={train_m['f1']:.4f} "
                        f"train_time={train_m['seconds']:.1f}s | "
                        f"val_loss={val_m['loss']:.4f} "
                        f"val_acc={val_m['acc']:.4f} "
                        f"val_f1={val_m['f1']:.4f} "
                        f"val_time={val_m['seconds']:.1f}s | "
                        f"lr={lr_now:.6f} "
                        f"model_eta={model_eta}"
                    )

            summary_row = {
                "dataset": name,
                "model": friendly_name,
                "origin": origin,
                "fold": fold,
                "best_epoch": best_val["epoch"],
                "best_acc": best_val["acc"],
                "best_prec": best_val["prec"],
                "best_rec": best_val["rec"],
                "best_spec": best_val["spec"],
                "best_f1": best_val["f1"],
                "best_ppv": best_val["ppv"],
                "best_npv": best_val["npv"],
            }
            append_csv_rows(summary_file, [summary_row], summary_cols)

            config_seconds = time.time() - config_t0
            config_end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""

            _append_results(
                {
                    "run_dir": str(run_dir),
                    "dataset": name,
                    "balance_mode": balance_mode,
                    "model": friendly_name,
                    "backbone_id": backbone_id,
                    "origin": origin,
                    "fold": fold,
                    "config_start": config_start_dt,
                    "config_end": config_end_dt,
                    "config_seconds": config_seconds,
                    "best_epoch": best_val["epoch"],
                    "best_acc": best_val["acc"],
                    "best_f1": best_val["f1"],
                    "device": str(device),
                    "gpu_name": gpu_name,
                    "amp": amp_flag,
                }
            )

            if callable(progress_cb):
                progress_cb(
                    dataset=name,
                    balance_mode=balance_mode,
                    model=friendly_name,
                    backbone_id=backbone_id,
                    fold=fold,
                    config_seconds=config_seconds,
                    run_dir=str(run_dir),
                )

            del model, optimiser, scheduler, train_loader, val_loader, scaler
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nFinished training on dataset {name}. Logs at: {run_dir}")

