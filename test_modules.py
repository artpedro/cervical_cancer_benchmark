from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets.datasets import (
    NORM_STATS_PATH,
    get_loaders,
    make_tf_from_stats_for_fold,
    scan_riva,
)
from model_loader import load_any


DEFAULT_MODELS = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "mobilenetv2_100",
    "mobilenet_v4",
]


def _take_one_batch(loader):
    it = iter(loader)
    images, labels = next(it)
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test RIVA dataloaders and model loading/forward."
    )
    parser.add_argument(
        "--riva-root",
        type=Path,
        default=Path("./datasets/data/riva"),
        help="Path to RIVA dataset root.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=NORM_STATS_PATH,
        help="Path to normalization_stats.json with riva fold stats.",
    )
    parser.add_argument("--dataset-name", type=str, default="riva")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Backbone aliases to test with model_loader.load_any.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name()}")

    if not args.riva_root.exists():
        raise FileNotFoundError(f"RIVA root not found: {args.riva_root}")
    if not args.stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_path}")

    print("\n[STEP] Scanning RIVA...")
    df = scan_riva(
        root=args.riva_root,
        num_folds=args.num_folds,
        seed=args.seed,
        test_size=args.test_size,
    )
    print(f"[OK] RIVA rows: {len(df)}")
    print(f"[OK] Split counts:\n{df['split'].value_counts().to_string()}")
    print(f"[OK] Class counts:\n{df['binary_label'].value_counts().to_string()}")

    print("\n[STEP] Building fold transforms and dataloaders...")
    train_tf, val_tf = make_tf_from_stats_for_fold(
        args.dataset_name, args.fold, args.stats_path
    )
    train_loader, val_loader = get_loaders(
        df=df,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        train_tf=train_tf,
        val_tf=val_tf,
    )
    train_x, train_y = _take_one_batch(train_loader)
    val_x, val_y = _take_one_batch(val_loader)
    print(
        f"[OK] Train batch: x={tuple(train_x.shape)} y={tuple(train_y.shape)} "
        f"labels={train_y.tolist()}"
    )
    print(
        f"[OK] Val batch:   x={tuple(val_x.shape)} y={tuple(val_y.shape)} "
        f"labels={val_y.tolist()}"
    )

    print("\n[STEP] Loading models and running one forward pass...")
    failures: list[str] = []
    sample = train_x.to(device, non_blocking=True)

    for name in args.models:
        try:
            model, in_features, origin, n_params = load_any(
                name=name,
                num_classes=2,
                pretrained=args.pretrained,
                device=device,
                max_params_m=100.0,
            )
            model = model.to(device)
            model.eval()

            with torch.inference_mode():
                y = model(sample)
                if isinstance(y, (tuple, list)):
                    y = y[0]

            if not isinstance(y, torch.Tensor):
                raise TypeError(f"{name}: forward returned {type(y)}")
            if y.ndim != 2 or y.shape[1] != 2:
                raise RuntimeError(f"{name}: expected logits shape [B,2], got {tuple(y.shape)}")

            print(
                f"[OK] {name:18s} origin={origin:40s} "
                f"in_features={in_features!s:>6s} params={n_params/1e6:.3f}M "
                f"out={tuple(y.shape)}"
            )
        except Exception as exc:
            msg = f"{name}: {exc}"
            failures.append(msg)
            print(f"[FAIL] {msg}")

    print("\n[SUMMARY]")
    if failures:
        for item in failures:
            print(f" - {item}")
        raise SystemExit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()

