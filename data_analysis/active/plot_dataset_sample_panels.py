from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from datasets.datasets import (
    NORM_STATS_PATH,
    make_tf_from_stats_full,
    scan_herlev,
    scan_riva,
    scan_sipakmed,
)


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("datasets/data")
OUT_DIR = Path("workspace/analysis/sample_visualization")

SEED = 42
NUM_FOLDS = 5
TEST_SIZE = 0.2
SPLIT = "train_dev"  # "test" or "train_dev"

DATASET_ORDER = ("herlev", "sipakmed", "riva")
CLASS_ORDER = ("normal", "abnormal")

# Randomized selection (deterministic with seed): number of images per class.
SAMPLES_PER_CLASS = 2
SAMPLE_SELECTION_SEED = 12343322212

# Transformed panel: match training (augmented) or evaluation (deterministic) pipeline.
# Training uses train_tf from datasets.datasets.make_tf (RandomRotation, flip, ColorJitter, ...).
# Eval/test uses eval_tf only (Resize, CenterCrop, Normalize).
TRANSFORM_MODE: str = "train"  # "train" | "eval"

# Fixed RNG per (dataset, class, sample index) so train augmentations are reproducible in the figure.
TRANSFORM_RNG_SEED_BASE = 12345

# Normalization mean/std here come from make_tf_from_stats_full (JSON key "full" / "train_dev").
# The training loop uses make_tf_from_stats_for_fold per CV fold; for pixel-identical norm to a given fold,
# you would need to swap in that helper with a chosen FOLD index (not implemented by default).

# For transformed panel:
# - True: undo normalization for human-readable RGB (still resize/crop equivalent).
# - False: show raw normalized tensor clipped to [0,1] (closer to model numeric input).
DENORMALIZE_FOR_DISPLAY = True

DPI = 300
RC_ACADEMIC: dict = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
}


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


def _scanner(dataset: str) -> Callable[..., pd.DataFrame]:
    if dataset == "herlev":
        return scan_herlev
    if dataset == "sipakmed":
        return scan_sipakmed
    if dataset == "riva":
        return scan_riva
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _dataset_display(dataset: str) -> str:
    return {"herlev": "Herlev", "sipakmed": "SIPaKMeD", "riva": "Riva"}.get(dataset, dataset)


def _class_display(cls: str) -> str:
    return {"normal": "Normal", "abnormal": "Abnormal"}.get(cls, cls)


def _denorm_from_stats(dataset: str, stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import json

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    entry = stats[dataset].get("full") or stats[dataset].get("train_dev")
    if entry is None:
        raise KeyError(
            f"Normalization stats for {dataset!r} not found under 'full' or 'train_dev' in {stats_path}."
        )
    mean = np.asarray(entry["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(entry["std"], dtype=np.float32).reshape(3, 1, 1)
    return mean, std


def _select_random_samples(
    df: pd.DataFrame,
    *,
    dataset: str,
    cls: str,
    n_samples: int,
    seed: int,
) -> list[pd.Series]:
    sub = (
        df[(df["split"] == SPLIT) & (df["binary_label"] == cls)]
        .sort_values("path")
        .reset_index(drop=True)
    )
    if sub.empty:
        raise RuntimeError(f"No samples for dataset={dataset}, split={SPLIT}, class={cls}.")
    if n_samples < 1:
        raise ValueError("SAMPLES_PER_CLASS must be >= 1")
    if n_samples > len(sub):
        raise ValueError(
            f"SAMPLES_PER_CLASS={n_samples} exceeds available samples={len(sub)} "
            f"for dataset={dataset}, class={cls}, split={SPLIT}."
        )
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(sub), size=n_samples, replace=False).astype(int))
    return [sub.iloc[int(i)] for i in idx]


def _apply_transform(tf: Callable, img: Image.Image, rng_seed: int) -> torch.Tensor:
    """Apply torchvision Compose; seed RNG so train augmentations are reproducible."""
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    return tf(img)


def _transform_rng_seed(dataset: str, cls: str, sample_k: int) -> int:
    di = DATASET_ORDER.index(dataset)
    ci = CLASS_ORDER.index(cls)
    return TRANSFORM_RNG_SEED_BASE + di * 10_000 + ci * 100 + sample_k


def _to_display_from_tensor(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.float32)
    if DENORMALIZE_FOR_DISPLAY:
        arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))
    return arr


def _plot_panel(
    selected: dict[str, dict[str, list[pd.Series]]],
    tf_by_ds: dict[str, Callable],
    mean_std_by_ds: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    n_ds = len(DATASET_ORDER)
    counts_by_class = {cls: int(SAMPLES_PER_CLASS) for cls in CLASS_ORDER}
    n_cols = sum(counts_by_class.values())
    col_keys: list[tuple[str, int]] = []
    for cls in CLASS_ORDER:
        for k in range(counts_by_class[cls]):
            col_keys.append((cls, k))

    with plt.rc_context(RC_ACADEMIC):
        # Build explicit rows: class-header row + [dataset-title, image-row] x dataset
        fig_raw = plt.figure(figsize=(2.15 * n_cols, 2.45 * n_ds + 0.8), constrained_layout=False)
        fig_tf = plt.figure(figsize=(2.15 * n_cols, 2.45 * n_ds + 0.8), constrained_layout=False)
        gs_raw = fig_raw.add_gridspec(
            nrows=1 + n_ds * 2,
            ncols=n_cols,
            height_ratios=[0.16] + [0.14 if i % 2 == 0 else 1.0 for i in range(n_ds * 2)],
            hspace=0.14,
            wspace=0.04,
        )
        gs_tf = fig_tf.add_gridspec(
            nrows=1 + n_ds * 2,
            ncols=n_cols,
            height_ratios=[0.16] + [0.14 if i % 2 == 0 else 1.0 for i in range(n_ds * 2)],
            hspace=0.14,
            wspace=0.04,
        )
        axes_raw = np.empty((n_ds, n_cols), dtype=object)
        axes_tf = np.empty((n_ds, n_cols), dtype=object)

        # Dedicated class-header band (never overlaps image rows).
        split_col = counts_by_class[CLASS_ORDER[0]]
        ax_cls_raw_n = fig_raw.add_subplot(gs_raw[0, :split_col])
        ax_cls_raw_a = fig_raw.add_subplot(gs_raw[0, split_col:])
        ax_cls_tf_n = fig_tf.add_subplot(gs_tf[0, :split_col])
        ax_cls_tf_a = fig_tf.add_subplot(gs_tf[0, split_col:])
        for ax, txt in (
            (ax_cls_raw_n, "Class: Normal"),
            (ax_cls_raw_a, "Class: Abnormal"),
            (ax_cls_tf_n, "Class: Normal"),
            (ax_cls_tf_a, "Class: Abnormal"),
        ):
            ax.axis("off")
            ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=13, color="0.12", fontweight="bold")

        for r, ds in enumerate(DATASET_ORDER):
            title_row = 1 + 2 * r
            img_row = title_row + 1

            # Dataset section title row (separate from image axes)
            ax_title_raw = fig_raw.add_subplot(gs_raw[title_row, :])
            ax_title_tf = fig_tf.add_subplot(gs_tf[title_row, :])
            for ax_t, ttl in (
                (ax_title_raw, f"{_dataset_display(ds)} ({SPLIT} split)"),
                (ax_title_tf, f"{_dataset_display(ds)} ({SPLIT} split)"),
            ):
                ax_t.axis("off")
                ax_t.text(
                    0.5,
                    0.35,
                    ttl,
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="0.12",
                    fontweight="bold",
                )

            tf_ds = tf_by_ds[ds]
            mean, std = mean_std_by_ds[ds]

            for c, (cls, k) in enumerate(col_keys):
                axr = fig_raw.add_subplot(gs_raw[img_row, c])
                axt = fig_tf.add_subplot(gs_tf[img_row, c])
                axes_raw[r, c] = axr
                axes_tf[r, c] = axt

                row = selected[ds][cls][k]
                path = Path(str(row["path"]))

                img = Image.open(path).convert("RGB")
                axr.imshow(img)
                axr.axis("off")

                rng_seed = _transform_rng_seed(ds, cls, k)
                x = _apply_transform(tf_ds, img, rng_seed)
                disp = _to_display_from_tensor(x, mean, std)
                axt.imshow(disp)
                axt.axis("off")

        fig_raw.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.93)
        fig_tf.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.93)

        fig_raw.suptitle("Raw image samples by class", fontsize=16, color="0.05", y=0.97, fontweight="bold")
        tf_note = "denormalized display" if DENORMALIZE_FOR_DISPLAY else "normalized tensor (clipped)"
        if TRANSFORM_MODE == "train":
            tf_caption = (
                f"Model input after training augmentations ({tf_note})"
            )
        else:
            tf_caption = f"Model input after evaluation transform ({tf_note})"
        fig_tf.suptitle(tf_caption, fontsize=16, color="0.05", y=0.97, fontweight="bold")

        OUT_DIR.resolve().mkdir(parents=True, exist_ok=True)
        out_raw = OUT_DIR / "raw_samples_grid.png"
        out_tf = OUT_DIR / "transformed_samples_grid.png"
        fig_raw.savefig(out_raw.resolve(), dpi=DPI, bbox_inches="tight")
        fig_tf.savefig(out_tf.resolve(), dpi=DPI, bbox_inches="tight")
        plt.close(fig_raw)
        plt.close(fig_tf)
    print(f"[OK] wrote {out_raw.resolve()}")
    print(f"[OK] wrote {out_tf.resolve()}")


def main() -> None:
    data_root = DATA_ROOT.resolve()
    stats_path = NORM_STATS_PATH.resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    frames: dict[str, pd.DataFrame] = {}
    tf_by_ds: dict[str, Callable] = {}
    mean_std_by_ds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ds in DATASET_ORDER:
        root = _resolve_dataset_root(data_root, ds)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found for {ds}: {root}")
        scan_fn = _scanner(ds)
        frames[ds] = scan_fn(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
        train_tf, eval_tf = make_tf_from_stats_full(ds, stats_path)
        if TRANSFORM_MODE not in ("train", "eval"):
            raise ValueError("TRANSFORM_MODE must be 'train' or 'eval'")
        tf_by_ds[ds] = train_tf if TRANSFORM_MODE == "train" else eval_tf
        mean_std_by_ds[ds] = _denorm_from_stats(ds, stats_path)

    selected: dict[str, dict[str, list[pd.Series]]] = {}
    for ds in DATASET_ORDER:
        selected[ds] = {}
        for cls in CLASS_ORDER:
            class_seed = SAMPLE_SELECTION_SEED + 1000 * DATASET_ORDER.index(ds) + CLASS_ORDER.index(cls)
            selected[ds][cls] = _select_random_samples(
                frames[ds],
                dataset=ds,
                cls=cls,
                n_samples=SAMPLES_PER_CLASS,
                seed=class_seed,
            )

    _plot_panel(selected, tf_by_ds, mean_std_by_ds)


if __name__ == "__main__":
    main()

