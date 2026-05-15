from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.datasets import (
    NORM_STATS_PATH,
    scan_herlev,
    scan_riva,
    scan_sipakmed,
)

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("datasets/data")
OUT_DIR = Path("workspace/analysis/transform_probe")

SEED = 42
NUM_FOLDS = 5
TEST_SIZE = 0.2
SPLIT = "train_dev"

DATASET_ORDER = ("herlev", "sipakmed", "riva")
SAMPLE_SELECTION_SEED = 12343322212
TRANSFORM_RNG_SEED_BASE = 12345

# 12 cells wide × 6 tall: each row is 6 (original | transformed) pairs → 36 unique images.
GRID_ROWS = 6
GRID_COLS = 12
N_IMAGES = GRID_ROWS * (GRID_COLS // 2)

DPI = 120

# Pre-transform column: fit image inside this square without stretching (letterbox padding).
LETTERBOX_SIZE = 224
LETTERBOX_FILL = (255, 255, 255)

# Undo Normalize(mean, std) before imshow (transformed column; must end with ToTensor + Normalize).
DENORMALIZE_FOR_DISPLAY = True


def experimental_tf_herlev(mean: list[float], std: list[float]) -> Callable:
    """Experimental augmentations for Herlev (edit independently of other datasets)."""
    return T.Compose(
        [
            T.Resize(224),
            T.RandomRotation(
                degrees=180,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
            ),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.25, 0.25, 0.25, 0.0),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def experimental_tf_sipakmed(mean: list[float], std: list[float]) -> Callable:
    """Experimental augmentations for SIPaKMeD."""
    return T.Compose(
        [
            T.Resize(224),
            T.RandomRotation(
                degrees=180,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
            ),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.25, 0.25, 0.25, 0.0),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def experimental_tf_riva(mean: list[float], std: list[float]) -> Callable:
    """Experimental augmentations for RIVA."""
    return T.Compose(
        [
            T.RandomRotation(
                degrees=180,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
            ),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.25, 0.25, 0.25, 0.0),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


# One entry per dataset in DATASET_ORDER (add a function + map entry if you extend the tuple).
EXPERIMENTAL_TF_BUILDERS: dict[str, Callable[[list[float], list[float]], Callable]] = {
    "herlev": experimental_tf_herlev,
    "sipakmed": experimental_tf_sipakmed,
    "riva": experimental_tf_riva,
}


def make_experimental_tf(dataset: str, mean: list[float], std: list[float]) -> Callable:
    try:
        builder = EXPERIMENTAL_TF_BUILDERS[dataset]
    except KeyError as e:
        raise KeyError(
            f"No experimental transform registered for dataset={dataset!r}. "
            "Add a builder and register it in EXPERIMENTAL_TF_BUILDERS."
        ) from e
    return builder(mean, std)


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


def _scanner(dataset: str):
    if dataset == "herlev":
        return scan_herlev
    if dataset == "sipakmed":
        return scan_sipakmed
    if dataset == "riva":
        return scan_riva
    raise ValueError(f"Unsupported dataset: {dataset!r}")


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


def _select_samples(df: pd.DataFrame, n: int, seed: int) -> list[pd.Series]:
    sub = df[df["split"] == SPLIT].sort_values("path").reset_index(drop=True)
    if sub.empty:
        raise RuntimeError(f"No samples for split={SPLIT}.")
    if n > len(sub):
        raise ValueError(f"Need {n} images but only {len(sub)} available for split={SPLIT}.")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(sub), size=n, replace=False).astype(int))
    return [sub.iloc[int(i)] for i in idx]


def _apply_transform(tf: Callable, img: Image.Image, rng_seed: int) -> torch.Tensor:
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    return tf(img)


def _to_display_tensor(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.float32)
    if DENORMALIZE_FOR_DISPLAY:
        arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    return np.transpose(arr, (1, 2, 0))


def _letterbox_rgb_display(img: Image.Image, size: int, fill: tuple[int, int, int]) -> np.ndarray:
    """HWC float32 in [0, 1], square `size`, original aspect ratio preserved (padded with `fill`)."""
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("Image has non-positive size.")
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _plot_dataset_grid(
    dataset: str,
    *,
    data_root: Path,
    stats_path: Path,
    dataset_index: int,
) -> Path:
    if GRID_COLS % 2 != 0:
        raise ValueError("GRID_COLS must be even (original | transformed pairs).")
    pairs_per_row = GRID_COLS // 2
    if pairs_per_row * GRID_ROWS != N_IMAGES:
        raise ValueError("N_IMAGES must equal GRID_ROWS * (GRID_COLS // 2).")

    root = _resolve_dataset_root(data_root, dataset)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    df = _scanner(dataset)(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    mean, std = _denorm_from_stats(dataset, stats_path)
    mean_list = mean.flatten().tolist()
    std_list = std.flatten().tolist()
    experimental_tf = make_experimental_tf(dataset, mean_list, std_list)

    pick_seed = SAMPLE_SELECTION_SEED + 10_000 * dataset_index
    samples = _select_samples(df, N_IMAGES, pick_seed)

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 0.9, GRID_ROWS * 0.9))
    axes = np.asarray(axes)
    if GRID_ROWS == 1:
        axes = axes.reshape(1, -1)
    elif GRID_COLS == 1:
        axes = axes.reshape(-1, 1)

    for r in range(GRID_ROWS):
        for p in range(pairs_per_row):
            i = r * pairs_per_row + p
            path = Path(str(samples[i]["path"]))
            img = Image.open(path).convert("RGB")
            col_orig = 2 * p
            col_tf = col_orig + 1
            rng_seed = TRANSFORM_RNG_SEED_BASE + 100_000 * dataset_index + i

            orig_display = _letterbox_rgb_display(img, LETTERBOX_SIZE, LETTERBOX_FILL)
            x1 = _apply_transform(experimental_tf, img, rng_seed + 1_000_000)

            axes[r, col_orig].imshow(orig_display)
            axes[r, col_tf].imshow(_to_display_tensor(x1, mean, std))
            axes[r, col_orig].axis("off")
            axes[r, col_tf].axis("off")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_grid_12x6.png"
    fig.savefig(out_path.resolve(), dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def main() -> None:
    data_root = DATA_ROOT.resolve()
    stats_path = NORM_STATS_PATH.resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    missing = [ds for ds in DATASET_ORDER if ds not in EXPERIMENTAL_TF_BUILDERS]
    if missing:
        raise KeyError(
            f"DATASET_ORDER has datasets without EXPERIMENTAL_TF_BUILDERS entries: {missing}"
        )

    for di, ds in enumerate(DATASET_ORDER):
        out = _plot_dataset_grid(ds, data_root=data_root, stats_path=stats_path, dataset_index=di)
        print(f"[OK] wrote {out.resolve()}")


if __name__ == "__main__":
    main()
