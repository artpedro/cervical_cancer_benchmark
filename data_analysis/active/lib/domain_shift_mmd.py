"""Maximum mean discrepancy (MMD) helpers for domain-shift analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import rbf_kernel

# Dataset-pair semantics aligned with RQ4 FID ordering (low -> high shift).
PAIR_LEVEL_BY_DATASETS: dict[frozenset[str], str] = {
    frozenset({"herlev", "sipakmed"}): "low",
    frozenset({"herlev", "riva"}): "high",
    frozenset({"riva", "sipakmed"}): "medium",
}
LEVEL_ORDER: tuple[str, ...] = ("low", "medium", "high")
LEVEL_LABELS: dict[str, str] = {"low": "Low", "medium": "Medium", "high": "High"}


def pair_level(dataset_a: str, dataset_b: str) -> str:
    key = frozenset({str(dataset_a), str(dataset_b)})
    if key not in PAIR_LEVEL_BY_DATASETS:
        raise KeyError(f"Unknown dataset pair: {dataset_a}, {dataset_b}")
    return PAIR_LEVEL_BY_DATASETS[key]


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray, *, max_samples: int = 500) -> float:
    """RBF bandwidth: 1 / median squared pairwise distance (subsampled for speed)."""
    rng = np.random.default_rng(0)
    z = np.concatenate([x, y], axis=0).astype(np.float64, copy=False)
    n = z.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        z = z[idx]
    if z.shape[0] < 2:
        return 1.0
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    tri = d2[np.triu_indices(d2.shape[0], k=1)]
    med = float(np.median(tri)) if tri.size else 1.0
    if med <= 0:
        med = 1.0
    return 1.0 / med


def mmd_rbf(x: np.ndarray, y: np.ndarray, *, gamma: float | None = None) -> float:
    """Unbiased MMD with RBF kernel; returns sqrt(max(MMD^2, 0))."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    if gamma is None:
        gamma = median_heuristic_gamma(x, y)
    n, m = x.shape[0], y.shape[0]
    kxx = rbf_kernel(x, x, gamma=gamma)
    kyy = rbf_kernel(y, y, gamma=gamma)
    kxy = rbf_kernel(x, y, gamma=gamma)
    mmd2 = (
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )
    return float(np.sqrt(max(mmd2, 0.0)))


def aggregate_mmd(
    components: Iterable[str],
    target: str,
    lookup: dict[tuple[str, str], float],
    *,
    aggregation: str,
) -> float:
    values: list[float] = []
    for component in components:
        c = str(component).strip()
        if not c or c == target:
            continue
        v = lookup.get((c, target))
        if v is None:
            v = lookup.get((target, c))
        if v is None:
            continue
        values.append(float(v))
    if not values:
        return float("nan")
    if aggregation == "min":
        return float(min(values))
    if aggregation == "max":
        return float(max(values))
    if aggregation == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"Unsupported aggregation={aggregation!r}")


def observation_mmd_level(
    components: Iterable[str],
    target: str,
    lookup: dict[tuple[str, str], float],
    *,
    aggregation: str,
) -> tuple[float, str]:
    """Return (mmd_to_target, shift level) using pair identity of the winning edge."""
    candidates: list[tuple[float, str]] = []
    for component in components:
        c = str(component).strip()
        if not c or c == target:
            continue
        v = lookup.get((c, target))
        if v is None:
            v = lookup.get((target, c))
        if v is None:
            continue
        candidates.append((float(v), pair_level(c, target)))
    if not candidates:
        return float("nan"), ""
    if aggregation == "min":
        return min(candidates, key=lambda item: item[0])
    if aggregation == "max":
        return max(candidates, key=lambda item: item[0])
    if aggregation == "mean":
        mmd = float(sum(v for v, _ in candidates) / len(candidates))
        # Level from the closest canonical pair median is ambiguous; use min edge level.
        return min(candidates, key=lambda item: item[0])
    raise ValueError(f"Unsupported aggregation={aggregation!r}")


def _pool_to_vector(feat: torch.Tensor, model: nn.Module) -> torch.Tensor:
    if feat.dim() == 2:
        return feat
    if hasattr(model, "global_pool"):
        pool = model.global_pool
        if callable(pool):
            feat = pool(feat)
            return feat.flatten(1) if feat.dim() > 2 else feat
        if isinstance(pool, str) and pool in {"avg", "avgmax"}:
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
            return feat.flatten(1)
    if feat.dim() == 4:
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
    return feat.flatten(1)


@torch.inference_mode()
def forward_embeddings(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Penultimate embeddings used as the model representation space."""
    if hasattr(model, "forward_features"):
        feat = model.forward_features(images)
        return _pool_to_vector(feat, model)

    if hasattr(model, "features"):
        feat = model.features(images)
        if hasattr(model, "avgpool"):
            feat = model.avgpool(feat)
        return _pool_to_vector(feat, model)

    if hasattr(model, "backbone"):
        feat = model.backbone(images)
        return _pool_to_vector(feat, model)

    # Torchvision ShuffleNetV2 / similar stacks without a ``features`` module.
    head_attr = next((a for a in ("fc", "classifier", "head") if hasattr(model, a)), None)
    if head_attr is not None:
        head = getattr(model, head_attr)
        in_dim = getattr(head, "in_features", None)
        if in_dim is not None:
            device = images.device
            dtype = head.weight.dtype
            setattr(model, head_attr, nn.Identity())
            try:
                feat = model(images)
            finally:
                setattr(model, head_attr, head)
            return _pool_to_vector(feat, model)

    raise RuntimeError(f"Cannot extract embeddings from {type(model).__name__}")
