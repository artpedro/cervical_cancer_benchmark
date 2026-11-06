"""
Per-image prediction export for protocol v2 (one file per fit, written right after the best checkpoint is selected).

Rows exported per fit:
  * every training source: validation fold (eval_role='val') and fixed holdout (eval_role='test')   -> in-domain
  * every dataset that is NOT a training source: all images (eval_role='ood'; split column keeps train_dev/test)
Normalization (one key per run, ``protocol_v2.norm_key``): every exported image of a run -- validation, holdout and unseen
target -- is normalised with the run's training-only statistics (solo: the source's own fold-k stats; mixed: the pooled
fold-k training stats of the two sources). Target-dataset statistics are never used.

Evaluation populations (each unseen-target image is exported ONCE; the two populations are nested):
  * eval_population == 'ood_target_holdout'     -> primary population: the target's fixed holdout (manuscript definition)
  * eval_population in ('ood_target_holdout','ood_target_development') -> supplementary: the entire unseen target dataset
  * 'source_val_fold' / 'source_holdout'        -> in-domain rows of each training source
``in_target_holdout`` is 1 for holdout images of any role.
"""
from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from datasets.datasets import PapDataset, make_tf
from datasets import protocol_v2 as P
from model_loader import load_any

COLUMNS = [
    "experiment_version", "run_name", "regime", "train_sources", "label_mapping", "endpoint_arm",
    "checkpoint_id", "checkpoint_sha256", "best_epoch", "backbone_id", "model", "origin", "fold",
    "target_dataset", "eval_role", "eval_population", "in_target_holdout", "split", "sample_id", "relpath", "group_id", "g_image", "label_full",
    "y_true_primary_metnormal", "y_true_sensitivity_metabnormal", "y_true_train_mapping",
    "logit_normal", "logit_abnormal", "p_abnormal", "pred_train_mapping_threshold_0p5",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def pooled_stats(stats: dict, bases: list[str], fold: int) -> tuple[list[float], list[float]]:
    """Pixel-weighted pooled mean/std of the fold-k training statistics of several datasets (equal pixels per image)."""
    n = np.array([stats[b][f"fold_{fold}"]["n_train"] for b in bases], float)
    m = np.array([stats[b][f"fold_{fold}"]["mean"] for b in bases])
    s = np.array([stats[b][f"fold_{fold}"]["std"] for b in bases])
    w = n / n.sum()
    mean = (w[:, None] * m).sum(0)
    ex2 = (w[:, None] * (s ** 2 + m ** 2)).sum(0)
    return mean.tolist(), np.sqrt(ex2 - mean ** 2).tolist()


def build_prediction_frame(meta: dict, *, backbone_id: str, model: str, origin: str, fold: int, ckpt_sha: str, best_epoch: int,
                           part: pd.DataFrame, target: str, role: str, logits: np.ndarray, train_col: str) -> pd.DataFrame:
    """Single place that defines the export schema (also used by the synthetic interface test)."""
    lg = np.asarray(logits, dtype=np.float64)
    z = lg - lg.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    cid = f"{meta['run_name']}__{backbone_id}__fold{fold}"
    is_hold = (part["split"].to_numpy() == "test")
    if role == "val":
        pop = np.full(len(part), "source_val_fold", dtype=object)
    elif role == "test":
        pop = np.full(len(part), "source_holdout", dtype=object)
    else:
        pop = np.where(is_hold, "ood_target_holdout", "ood_target_development").astype(object)
    df = pd.DataFrame({
        "experiment_version": meta["experiment_version"], "run_name": meta["run_name"], "regime": meta["regime"],
        "train_sources": meta["train_sources"], "label_mapping": meta["label_mapping"], "endpoint_arm": meta["endpoint_arm"],
        "checkpoint_id": cid, "checkpoint_sha256": ckpt_sha, "best_epoch": int(best_epoch), "backbone_id": backbone_id, "model": model, "origin": origin,
        "fold": fold, "target_dataset": target, "eval_role": role, "eval_population": pop, "in_target_holdout": is_hold.astype(int), "split": part["split"].to_numpy(),
        "sample_id": part["sample_id"].to_numpy(), "relpath": part["relpath"].to_numpy(),
        "group_id": part["group_id"].to_numpy(),
        "g_image": part["g_image"].to_numpy() if "g_image" in part else part["group_id"].to_numpy(),
        "label_full": part["label_full"].to_numpy(),
        "y_true_primary_metnormal": part["binary_idx_metnormal"].to_numpy(),
        "y_true_sensitivity_metabnormal": part["binary_idx"].to_numpy(),
        "y_true_train_mapping": part[train_col].to_numpy() if train_col else part["binary_idx_metnormal"].to_numpy(),
        "logit_normal": lg[:, 0], "logit_abnormal": lg[:, 1], "p_abnormal": p[:, 1],
        "pred_train_mapping_threshold_0p5": (p[:, 1] > 0.5).astype(int),
    })
    return df[COLUMNS]


@torch.no_grad()
def _logits(model, df: pd.DataFrame, tf, *, batch_size: int, num_workers: int, device, use_amp: bool) -> np.ndarray:
    d = df.copy(); d["binary_idx"] = d["binary_idx"].astype(int)
    out = []
    for x, _ in DataLoader(PapDataset(d, tf), batch_size=batch_size, shuffle=False, num_workers=num_workers):
        with autocast("cuda", enabled=bool(use_amp and device.type == "cuda")):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def eval_sets(bases: list[str], fold: int, ood_scope: str = "full") -> list[tuple[str, str, pd.DataFrame]]:
    """(target dataset, eval_role, partition rows) exported for a fit whose training sources are ``bases``."""
    out = []
    for target in ("herlev", "sipakmed", "riva"):
        part = pd.read_csv(P.PART_DIR / f"{target}_partition.csv")
        part["path"] = [str(P.ROOTS[target] / r) for r in part.relpath]
        part["binary_idx"] = part["binary_idx"].astype(int)
        if target in bases:
            for role, sel in (("val", (part.split == "train_dev") & (part.fold == fold)), ("test", part.split == "test")):
                out.append((target, role, part[sel].reset_index(drop=True)))
        else:
            sub = part if ood_scope == "full" else part[part.split == "test"]
            out.append((target, "ood", sub.reset_index(drop=True)))
    return out


def export_fit_predictions(*, run_name: str, backbone_id: str, display_name: str, origin: str, fold: int,
                           best_ckpt_path: Path, model_cfg: Any, device: torch.device, use_amp: bool,
                           out_dir: Path, batch_size: int = 64, num_workers: int = 8,
                           stats_path: Path = P.STATS_V2_PATH, ood_scope: str = "full") -> Path:
    meta = P.run_meta(run_name)
    srcs = P.run_sources(run_name)
    bases = [P.base_dataset(s) for s in srcs]
    stats = json.loads(Path(stats_path).read_text())
    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    model, _, _, _ = load_any(backbone_id, num_classes=2, pretrained=False, device=device,
                              max_params_m=model_cfg.max_params_m, **model_cfg.load_kwargs)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    sha = sha256_file(Path(best_ckpt_path))
    # label column that defines y_true under THIS run's training mapping (SIPaKMeD only differs)
    train_col = "binary_idx_metnormal" if meta["label_mapping"] == P.PRIMARY_MAPPING else "binary_idx"

    frames = []
    nk = P.norm_key(run_name)
    m_, s_ = stats[nk][f"fold_{fold}"]["mean"], stats[nk][f"fold_{fold}"]["std"]    # same statistics for every exported image
    for target, role, sub in eval_sets(bases, fold, ood_scope):
        tf = make_tf(normalizing_matriz=[m_, s_])[1]
        lg = _logits(model, sub, tf, batch_size=batch_size, num_workers=num_workers, device=device, use_amp=use_amp)
        frames.append(build_prediction_frame(meta, backbone_id=backbone_id, model=display_name, origin=origin, fold=fold,
                                             ckpt_sha=sha, best_epoch=int(ckpt['epoch']), part=sub, target=target, role=role, logits=lg, train_col=train_col))
    out = Path(out_dir) / run_name / f"{backbone_id}__fold{fold}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")                       # atomic: never leave a partial prediction file
    pd.concat(frames, ignore_index=True).to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out)
    del model; torch.cuda.empty_cache()
    return out
