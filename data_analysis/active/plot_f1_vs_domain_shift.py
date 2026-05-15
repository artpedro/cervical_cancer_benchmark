from __future__ import annotations

"""
Plot cross-dataset F1 vs domain shift (FID) between dataset pairs.

1. Compute Fréchet Inception Distance (FID) between every ordered pair of
   {herlev, sipakmed, riva} using InceptionV3 pool3 features on the test split.
   FID is symmetric so we compute 3 unordered pairs and use them for both directions.
2. Merge with the cross-dataset evaluation CSV (source→target mean F1 per model).
3. Scatter: x = FID(source, target), y = mean cross-dataset F1, one point per
   (model, source, target) triple, colored by CNN vs Transformer.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset

from datasets.datasets import (
    NORM_STATS_PATH,
    scan_herlev,
    scan_riva,
    scan_sipakmed,
)
from data_analysis.active.dataset_regime_utils import SOLO_DATASETS, is_mixed_dataset

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("datasets/data")
ANALYSIS_DIR = Path("workspace/analysis")
CROSS_CSV = ANALYSIS_DIR / "test_eval_results_cross_dataset" / "plots" / "test_split__aggregated_mean_by_source_target.csv"
IN_DOMAIN_CSV = ANALYSIS_DIR / "test_eval_results_cross_dataset" / "plots" / "test_split__in_domain_vs_mean_cross_dataset_summary.csv"
OUT_DIR = ANALYSIS_DIR / "domain_shift"

SEED = 42
TEST_SIZE = 0.2
NUM_FOLDS = 5
BATCH_SIZE = 64
NUM_WORKERS = 4
MAX_IMAGES_PER_DATASET = None  # None = use all test images
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DATASETS: tuple[str, ...] | None = SOLO_DATASETS
MIXED_ROW_POLICY = "exclude"  # exclude | keep

COLOR_CNN = "#1f77b4"
COLOR_TRANSFORMER = "#d62728"

MODEL_SHORT_NAMES: dict[str, str] = {
    "EfficientNet B0": "EN-B0",
    "EfficientNet B1": "EN-B1",
    "EfficientNet B2": "EN-B2",
    "MobileNet V2": "MBNetV2",
    "MobileNet V4": "MBNetV4",
    "EfficientFormerV2 S0": "EffF-S0",
    "EfficientFormerV2 S1": "EffF-S1",
    "MobileViT v2 100": "MViTv2-100",
    "FastViT T8": "FastViT-T8",
    "iformer_m": "iFormer-M",
}


# ============================================================
# Helpers
# ============================================================
def _model_family(model_name: str) -> str:
    name = str(model_name).lower()
    transformer_tokens = ("vit", "former", "iformer", "levit", "fastvit", "mobilevit", "eat")
    if any(tok in name for tok in transformer_tokens):
        return "transformer"
    return "cnn"


def _short(name: str) -> str:
    return MODEL_SHORT_NAMES.get(name, name)


def _normalize_cross_df(cross: pd.DataFrame) -> pd.DataFrame:
    out = cross.copy()
    # Older aggregated file uses source_label format: "<dataset> -> <model>"
    if "source_label" in out.columns and (
        "source_dataset" not in out.columns or "source_model" not in out.columns
    ):
        parts = out["source_label"].astype(str).str.split("->", n=1, expand=True)
        if parts.shape[1] == 2:
            out["source_dataset"] = parts[0].astype(str).str.strip()
            out["source_model"] = parts[1].astype(str).str.strip()
    if "target_dataset" not in out.columns and "target" in out.columns:
        out["target_dataset"] = out["target"].astype(str)
    if "mean_f1" not in out.columns and "target_f1_mean" in out.columns:
        out["mean_f1"] = out["target_f1_mean"]
    required = {"source_dataset", "target_dataset", "source_model", "mean_f1"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Cross CSV missing required columns after normalization: {missing}")
    return out


def _resolve_root(ds: str) -> Path:
    if ds == "herlev":
        return DATA_ROOT / "smear2005"
    if ds == "sipakmed":
        return DATA_ROOT / "sipakmed"
    if ds == "riva":
        low = DATA_ROOT / "riva"
        up = DATA_ROOT / "RIVA"
        return low if low.exists() else up
    raise ValueError(ds)


def _scan(ds: str, root: Path) -> pd.DataFrame:
    if ds == "herlev":
        return scan_herlev(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    if ds == "sipakmed":
        return scan_sipakmed(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    if ds == "riva":
        return scan_riva(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    raise ValueError(ds)


# ============================================================
# FID computation
# ============================================================
class _ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


def _inception_feature_extractor(device: torch.device) -> nn.Module:
    model = tvm.inception_v3(weights=tvm.Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


_FID_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.inference_mode()
def _extract_features(
    model: nn.Module,
    paths: list[str],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    ds = _ImagePathDataset(paths, _FID_TF)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def _compute_fid(feat1: np.ndarray, feat2: np.ndarray) -> float:
    mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
    mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


def _get_test_paths(ds: str) -> list[str]:
    root = _resolve_root(ds).resolve()
    df = _scan(ds, root)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    paths = test_df["path"].astype(str).tolist()
    if MAX_IMAGES_PER_DATASET is not None and len(paths) > MAX_IMAGES_PER_DATASET:
        rng = np.random.default_rng(SEED)
        paths = list(rng.choice(paths, size=MAX_IMAGES_PER_DATASET, replace=False))
    return paths


def compute_all_fid_pairs(device: torch.device) -> pd.DataFrame:
    print("[FID] Loading InceptionV3 …")
    inc = _inception_feature_extractor(device)

    features: dict[str, np.ndarray] = {}
    datasets = list(DATASETS) if DATASETS else list(SOLO_DATASETS)
    for ds in datasets:
        print(f"[FID] Extracting features for {ds} …")
        paths = _get_test_paths(ds)
        features[ds] = _extract_features(inc, paths, device=device, batch_size=BATCH_SIZE)
        print(f"  → {features[ds].shape[0]} images, {features[ds].shape[1]}-d features")

    rows = []
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            fid = _compute_fid(features[a], features[b])
            rows.append({"dataset_a": a, "dataset_b": b, "fid": fid})
            print(f"[FID] {a} ↔ {b}: {fid:.2f}")
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================
def _plot_f1_vs_fid(
    merged: pd.DataFrame,
    fid_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.set_axisbelow(True)
    for fam, color in [("cnn", COLOR_CNN), ("transformer", COLOR_TRANSFORMER)]:
        sub = merged[merged["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(sub["fid"], sub["mean_f1"], s=38, alpha=0.72, c=color, edgecolors="white",
                   linewidths=0.5, label=fam.upper(), zorder=3)

    from adjustText import adjust_text
    texts = []
    for _, r in merged.iterrows():
        txt = ax.text(
            r["fid"], r["mean_f1"],
            f"  {_short(r['source_model'])}",
            fontsize=5.5, alpha=0.85,
            color=COLOR_CNN if _model_family(r["source_model"]) == "cnn" else COLOR_TRANSFORMER,
        )
        texts.append(txt)
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4),
                force_text=(0.4, 0.6), expand=(1.15, 1.25))

    ax.set_xlabel("FID (domain shift magnitude)")
    ax.set_ylabel("Cross-dataset mean F1")
    ax.set_title("Cross-dataset F1 vs Domain Shift (FID)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    out_path = out_dir / "f1_vs_domain_shift_fid.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def _plot_f1_vs_fid_by_direction(
    merged: pd.DataFrame,
    out_dir: Path,
) -> None:
    """One panel per (source→target) direction; within each, scatter of models."""
    directions = merged.groupby(["source_dataset", "target_dataset"]).size().reset_index()
    n = len(directions)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows),
                             constrained_layout=True, squeeze=False)
    ax_flat = axes.ravel()

    for idx, (_, row) in enumerate(directions.iterrows()):
        src, tgt = row["source_dataset"], row["target_dataset"]
        ax = ax_flat[idx]
        sub = merged[(merged["source_dataset"] == src) & (merged["target_dataset"] == tgt)]
        fid_val = sub["fid"].iloc[0] if not sub.empty else 0
        for fam, color in [("cnn", COLOR_CNN), ("transformer", COLOR_TRANSFORMER)]:
            d = sub[sub["family"] == fam]
            if d.empty:
                continue
            ax.barh(d["source_model"].map(_short), d["mean_f1"], color=color, alpha=0.78, edgecolor="white",
                    linewidth=0.5, label=fam.upper())
        ax.set_title(f"{src} → {tgt}\nFID = {fid_val:.1f}", fontsize=9)
        ax.set_xlabel("Mean F1")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.invert_yaxis()

    for idx in range(n, len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig.suptitle("Cross-dataset F1 by transfer direction (sorted by FID)", fontsize=11)
    out_path = out_dir / "f1_by_transfer_direction_fid.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    device = torch.device(DEVICE)
    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fid_csv = out_dir / "pairwise_fid.csv"
    if fid_csv.exists():
        print(f"[FID] Loading cached {fid_csv}")
        fid_df = pd.read_csv(fid_csv)
    else:
        fid_df = compute_all_fid_pairs(device)
        fid_df.to_csv(fid_csv, index=False)
        print(f"[OK] wrote {fid_csv}")

    # Build symmetric lookup: (a,b) and (b,a) get same FID.
    fid_lookup: dict[tuple[str, str], float] = {}
    for _, r in fid_df.iterrows():
        fid_lookup[(r["dataset_a"], r["dataset_b"])] = r["fid"]
        fid_lookup[(r["dataset_b"], r["dataset_a"])] = r["fid"]

    cross_path = (CROSS_CSV if CROSS_CSV.is_absolute() else _REPO_ROOT / CROSS_CSV).resolve()
    if not cross_path.exists():
        raise FileNotFoundError(f"Missing cross-dataset CSV: {cross_path}")
    cross = _normalize_cross_df(pd.read_csv(cross_path))
    if MIXED_ROW_POLICY == "exclude":
        cross = cross[
            ~cross["source_dataset"].astype(str).map(is_mixed_dataset)
            & ~cross["target_dataset"].astype(str).map(is_mixed_dataset)
        ].copy()
    cross["fid"] = cross.apply(
        lambda r: fid_lookup.get((r["source_dataset"], r["target_dataset"]), float("nan")),
        axis=1,
    )
    cross["family"] = cross["source_model"].map(_model_family)
    cross = cross.dropna(subset=["fid", "mean_f1"]).copy()

    merged_csv = out_dir / "cross_f1_with_fid.csv"
    cross.sort_values(["fid", "source_model"]).to_csv(merged_csv, index=False)
    print(f"[OK] wrote {merged_csv}")

    _plot_f1_vs_fid(cross, fid_df, out_dir)
    _plot_f1_vs_fid_by_direction(cross, out_dir)

    print(f"[DONE] Domain shift outputs in: {out_dir}")


if __name__ == "__main__":
    main()
