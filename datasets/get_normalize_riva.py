from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.datasets import (
    NORM_STATS_PATH,
    compute_mean_std_for_df,
    scan_riva,
)


def _load_stats(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def compute_and_merge_riva_stats(
    *,
    root: Path,
    stats_path: Path,
    dataset_name: str = "riva",
    num_folds: int = 5,
    seed: int = 42,
    test_size: float = 0.2,
    max_samples: int | None = None,
) -> None:
    """
    Compute RIVA normalization stats and merge into existing JSON.

    Writes keys:
      - stats[dataset_name]["train_dev"]
      - stats[dataset_name]["full"]      (same subset as train_dev for this workflow)
      - stats[dataset_name]["fold_i"]    (i in 0..num_folds-1), each from that fold's train split
    """
    df = scan_riva(
        root=root,
        num_folds=num_folds,
        seed=seed,
        test_size=test_size,
    )

    train_dev_df = df[df["split"] == "train_dev"].reset_index(drop=True)
    if train_dev_df.empty:
        raise RuntimeError("RIVA train_dev split is empty; cannot compute normalization stats.")

    stats = _load_stats(stats_path)
    stats.setdefault(dataset_name, {})

    # train_dev / full stats
    mean_td, std_td = compute_mean_std_for_df(
        train_dev_df,
        max_samples=max_samples,
        show_progress=True,
    )
    stats[dataset_name]["train_dev"] = {"mean": mean_td, "std": std_td}
    stats[dataset_name]["full"] = {"mean": mean_td, "std": std_td}

    # Per-fold stats computed from each fold's training portion (no leakage)
    for fold in range(num_folds):
        fold_train_df = train_dev_df[train_dev_df["fold"] != fold].reset_index(drop=True)
        if fold_train_df.empty:
            raise RuntimeError(f"Fold {fold} training subset is empty.")

        mean_f, std_f = compute_mean_std_for_df(
            fold_train_df,
            max_samples=max_samples,
            show_progress=True,
        )
        stats[dataset_name][f"fold_{fold}"] = {"mean": mean_f, "std": std_f}

    _save_stats(stats_path, stats)
    print(f"[OK] Updated stats file: {stats_path}")
    print(f"[OK] Dataset key: {dataset_name}")
    print(f"[OK] Added keys: train_dev, full, fold_0..fold_{num_folds - 1}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and merge RIVA normalization stats into normalization_stats.json"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./datasets/data/riva"),
        help="Path to RIVA root folder",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=NORM_STATS_PATH,
        help="Path to normalization_stats.json",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="riva",
        help="Dataset key in the stats JSON",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for number of images used in each mean/std computation",
    )
    args = parser.parse_args()

    compute_and_merge_riva_stats(
        root=args.root,
        stats_path=args.stats_path,
        dataset_name=args.dataset_name,
        num_folds=args.num_folds,
        seed=args.seed,
        test_size=args.test_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

