"""Compute pairwise MMD in each model's representation space (test split).

Writes ``workspace/analysis/domain_shift/pairwise_mmd_by_model.csv`` with columns:
``source_model``, ``backbone_id``, ``dataset_a``, ``dataset_b``, ``mmd``.

Used by RQ6 (same design as RQ4, replacing FID with model-space MMD).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_analysis.active.lib.dataset_regime_utils import SOLO_DATASETS
from data_analysis.active.lib.domain_shift_mmd import mmd_rbf, forward_embeddings
from data_analysis.active.plots.plot_f1_vs_domain_shift import _get_test_paths
from model_loader import load_any
from train_models import ALL_MODEL_CONFIGS

OUT_DIR = Path("workspace/analysis/domain_shift")
OUT_CSV = OUT_DIR / "pairwise_mmd_by_model.csv"

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_IMAGES_PER_DATASET: int | None = None
DATASETS: tuple[str, ...] = SOLO_DATASETS

DISPLAY_TO_CONFIG = {cfg.display_name: cfg for cfg in ALL_MODEL_CONFIGS}


class _ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def _eval_transform(model: nn.Module, backbone_id: str) -> T.Compose:
    if hasattr(model, "pretrained_cfg") and model.pretrained_cfg:
        cfg = timm.data.resolve_model_data_config(model)
        return timm.data.create_transform(**cfg, is_training=False)
    if backbone_id == "eat":
        size = 224
    else:
        size = 224
    return T.Compose(
        [
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@torch.inference_mode()
def _extract_features(
    model: nn.Module,
    paths: list[str],
    transform,
    *,
    device: torch.device,
) -> np.ndarray:
    ds = _ImagePathDataset(paths, transform)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        emb = forward_embeddings(model, batch)
        chunks.append(emb.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _models_to_compute() -> list[tuple[str, str, dict]]:
    master = _REPO_ROOT / "workspace/analysis/generalizability/generalization_master.csv"
    if master.exists():
        names = sorted(pd.read_csv(master, usecols=["source_model"])["source_model"].astype(str).unique())
    else:
        names = sorted(DISPLAY_TO_CONFIG.keys())
    out: list[tuple[str, str, dict]] = []
    for name in names:
        cfg = DISPLAY_TO_CONFIG.get(name)
        if cfg is None:
            print(f"[WARN] No backbone config for {name!r}; skipping.")
            continue
        out.append((name, cfg.backbone_id, dict(cfg.load_kwargs)))
    return out


def compute_pairwise_mmd_for_model(
    display_name: str,
    backbone_id: str,
    load_kwargs: dict,
    *,
    device: torch.device,
    datasets: tuple[str, ...],
) -> list[dict[str, object]]:
    print(f"[MMD] {display_name} ({backbone_id}) …")
    model, _, _, _ = load_any(
        backbone_id,
        num_classes=2,
        pretrained=True,
        device=device,
        max_params_m=200.0,
        **load_kwargs,
    )
    model.eval()
    model.to(device)
    transform = _eval_transform(model, backbone_id)

    features: dict[str, np.ndarray] = {}
    for ds in datasets:
        paths = _get_test_paths(ds)
        if MAX_IMAGES_PER_DATASET is not None and len(paths) > MAX_IMAGES_PER_DATASET:
            rng = np.random.default_rng(SEED)
            paths = list(rng.choice(paths, size=MAX_IMAGES_PER_DATASET, replace=False))
        print(f"  [{display_name}] {ds}: {len(paths)} images")
        features[ds] = _extract_features(model, paths, transform, device=device)

    rows: list[dict[str, object]] = []
    for i, a in enumerate(datasets):
        for b in datasets[i + 1 :]:
            mmd = mmd_rbf(features[a], features[b])
            rows.append(
                {
                    "source_model": display_name,
                    "backbone_id": backbone_id,
                    "dataset_a": a,
                    "dataset_b": b,
                    "mmd": mmd,
                }
            )
            print(f"  [{display_name}] {a} ↔ {b}: MMD={mmd:.6f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    device = torch.device(DEVICE)
    out_dir = (_REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / OUT_CSV.name

    datasets = tuple(DATASETS)
    all_rows: list[dict[str, object]] = []
    for display_name, backbone_id, load_kwargs in _models_to_compute():
        try:
            all_rows.extend(
                compute_pairwise_mmd_for_model(
                    display_name,
                    backbone_id,
                    load_kwargs,
                    device=device,
                    datasets=datasets,
                )
            )
        except Exception as exc:
            print(f"[ERROR] {display_name}: {exc}")

    if not all_rows:
        raise RuntimeError("No pairwise MMD rows computed.")

    df = pd.DataFrame(all_rows).sort_values(["source_model", "dataset_a", "dataset_b"])
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
