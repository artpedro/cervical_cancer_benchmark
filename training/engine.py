from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

EPS = 1e-9


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    freq = df["binary_idx"].value_counts(normalize=True).sort_index()
    if freq.size != 2:
        raise ValueError("Expected two classes, got " + str(freq.to_dict()))
    return torch.tensor(
        [1.0 / freq[0], 1.0 / freq[1]], dtype=torch.float32, device=device
    )


def run_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    split_name: str,
    optimiser: Optional[torch.optim.Optimizer] = None,
    *,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    device: torch.device,
) -> dict[str, float]:
    training = optimiser is not None
    model.train(training)
    epoch_t0 = time.time()

    amp_enabled = bool(use_amp and torch.cuda.is_available() and device.type == "cuda")
    total_loss = 0.0
    n_samples = 0

    tp = torch.zeros((), device=device, dtype=torch.int64)
    tn = torch.zeros((), device=device, dtype=torch.int64)
    fp = torch.zeros((), device=device, dtype=torch.int64)
    fn = torch.zeros((), device=device, dtype=torch.int64)

    pbar = tqdm(
        dataloader,
        desc="train" if training else split_name,
        leave=False,
        dynamic_ncols=True,
    )

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            bs = labels.size(0)
            n_samples += bs

            if training:
                optimiser.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if amp_enabled:
                    if scaler is None:
                        raise ValueError("AMP enabled but scaler is None.")
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    optimiser.step()

            total_loss += float(loss.detach().item()) * bs
            pred = logits.detach().argmax(dim=1)

            tp += ((pred == 1) & (labels == 1)).sum().to(torch.int64)
            tn += ((pred == 0) & (labels == 0)).sum().to(torch.int64)
            fp += ((pred == 1) & (labels == 0)).sum().to(torch.int64)
            fn += ((pred == 0) & (labels == 1)).sum().to(torch.int64)

            running_loss = total_loss / max(1, n_samples)
            pbar.set_postfix(loss=f"{running_loss:.4f}")

    tp_f = tp.to(torch.float32)
    tn_f = tn.to(torch.float32)
    fp_f = fp.to(torch.float32)
    fn_f = fn.to(torch.float32)

    acc = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f + EPS)
    prec = tp_f / (tp_f + fp_f + EPS)
    rec = tp_f / (tp_f + fn_f + EPS)
    spec = tn_f / (tn_f + fp_f + EPS)
    f1 = (2.0 * prec * rec) / (prec + rec + EPS)
    ppv = prec
    npv = tn_f / (tn_f + fn_f + EPS)

    return {
        "loss": total_loss / max(1, n_samples),
        "acc": float(acc.item()),
        "prec": float(prec.item()),
        "rec": float(rec.item()),
        "spec": float(spec.item()),
        "f1": float(f1.item()),
        "ppv": float(ppv.item()),
        "npv": float(npv.item()),
        "seconds": time.time() - epoch_t0,
    }

