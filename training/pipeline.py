from __future__ import annotations

import datetime
import gc
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR

from datasets.datasets import (
    get_loader_mixed_eval,
    get_loaders_mixed,
    get_loaders,
    make_tf_from_stats_for_fold,
)
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
        "bal_acc",
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
                            bal_acc=m["bal_acc"],
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


def train_mixed_dataset_v2(
    name: str,
    df: pd.DataFrame,
    source_names: tuple[str, str],
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
    """
    Train on the union of two datasets' train_dev folds (same fold index for both sources),
    validate on the union of both val folds, and evaluate on the union of both test splits
    (plus per-source test F1). Uses per-row ``source_dataset`` and per-source augmentations
    / normalization from ``make_tf_from_stats_for_fold``.
    """
    assert balance_mode == "weighted_loss"
    if "source_dataset" not in df.columns:
        raise ValueError("train_mixed_dataset_v2 expects df['source_dataset']")
    seen = set(df["source_dataset"].astype(str).unique())
    expected = {source_names[0], source_names[1]}
    if seen != expected:
        raise ValueError(f"source_dataset values {seen!r} must match {expected!r}")

    print(f"\n{'=' * 70}")
    print(
        f"DATASET (MIXED): {name.upper()}  |  sources={source_names}  |  "
        f"balance_mode = {balance_mode}"
    )
    print(f"Run directory: {run_dir}")
    print(f"{'=' * 70}")

    train_df = df[df["split"].isin(["train", "train_dev"])].reset_index(drop=True)
    print(
        f"Using rows with split in ('train','train_dev') for training pipeline "
        f"(rows: {len(train_df)})"
    )

    class_dist = train_df["binary_idx"].value_counts().sort_index()
    total = len(train_df)
    print("\nClass Distribution (training subset, both sources):")
    print(
        f"  Class 0 (Normal):   {class_dist.get(0, 0):>6} samples ({100 * class_dist.get(0, 0) / total:5.1f}%)"
    )
    if 1 in class_dist.index:
        print(
            f"  Class 1 (Abnormal): {class_dist.get(1, 0):>6} samples ({100 * class_dist.get(1, 0) / total:5.1f}%)"
        )

    print("\n→ CrossEntropyLoss(class_weights) on mixed batches; per-source train/eval transforms.")
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
        "bal_acc",
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
        "test_f1_combined",
        "test_acc_combined",
        f"test_f1_{source_names[0]}",
        f"test_f1_{source_names[1]}",
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

            train_tf_by_source: dict[str, Any] = {}
            eval_tf_by_source: dict[str, Any] = {}
            for ds_key in source_names:
                t_tr, t_ev = make_tf_from_stats_for_fold(ds_key, fold, stats_path)
                train_tf_by_source[ds_key] = t_tr
                eval_tf_by_source[ds_key] = t_ev

            config_start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config_t0 = time.time()

            print(
                f"\n> {name} | {friendly_name} | fold {fold} "
                f"| model progress: "
                f"{eta_tracker.train_steps_done}/{eta_tracker.total_train_steps} train "
                f"+ {eta_tracker.val_steps_done}/{eta_tracker.total_val_steps} val "
                f"| ETA {eta_tracker.eta_hms()}"
            )

            pin = torch.cuda.is_available()
            train_loader, val_loader = get_loaders_mixed(
                df=train_df,
                fold=fold,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin,
                train_tf_by_source=train_tf_by_source,
                eval_tf_by_source=eval_tf_by_source,
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
                            bal_acc=m["bal_acc"],
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

            # --- Test evaluation (both sources, combined + per-source) ---
            test_df_full = df[df["split"] == "test"].reset_index(drop=True)
            if best_ckpt_path.exists():
                try:
                    ckpt = torch.load(
                        best_ckpt_path, map_location=device, weights_only=False
                    )
                except TypeError:
                    ckpt = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])
            model.eval()

            if len(test_df_full) > 0:
                test_loader_all = get_loader_mixed_eval(
                    test_df_full,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin,
                    eval_tf_by_source=eval_tf_by_source,
                )
                test_combined = run_epoch(
                    test_loader_all,
                    model,
                    criterion,
                    "test",
                    optimiser=None,
                    scaler=None,
                    use_amp=use_amp,
                    device=device,
                )
                print(
                    f"  Test (combined): acc={test_combined['acc']:.4f} f1={test_combined['f1']:.4f}"
                )
            else:
                test_combined = {k: float("nan") for k in ["acc", "f1"]}

            per_src_f1: dict[str, float] = {}
            for ds_key in source_names:
                sub = test_df_full[test_df_full["source_dataset"] == ds_key]
                if len(sub) == 0:
                    per_src_f1[ds_key] = float("nan")
                    continue
                tl = get_loader_mixed_eval(
                    sub,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin,
                    eval_tf_by_source=eval_tf_by_source,
                )
                tm = run_epoch(
                    tl,
                    model,
                    criterion,
                    f"test_{ds_key}",
                    optimiser=None,
                    scaler=None,
                    use_amp=use_amp,
                    device=device,
                )
                per_src_f1[ds_key] = tm["f1"]
                print(f"  Test ({ds_key}): acc={tm['acc']:.4f} f1={tm['f1']:.4f}")

            best_json_payload = {}
            if fold_best_json.exists():
                with open(fold_best_json, encoding="utf-8") as jf:
                    best_json_payload = json.load(jf)
            best_json_payload["test_combined"] = test_combined if len(test_df_full) > 0 else None
            best_json_payload["test_f1_per_source"] = per_src_f1
            write_json(fold_best_json, best_json_payload)

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
                "test_f1_combined": test_combined.get("f1", float("nan")),
                "test_acc_combined": test_combined.get("acc", float("nan")),
                f"test_f1_{source_names[0]}": per_src_f1.get(source_names[0], float("nan")),
                f"test_f1_{source_names[1]}": per_src_f1.get(source_names[1], float("nan")),
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
                    "test_f1_combined": summary_row["test_f1_combined"],
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

    print(f"\nFinished mixed training for {name}. Logs at: {run_dir}")

