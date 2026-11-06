"""
Revised protocol (v3) training entry point: 5 folds, persisted partitions, 9 run configurations x 16 models x 5 folds = 720 fits.

Run configurations (all read the frozen partitions in datasets/partitions_v3_5fold/, never re-scan the raw folders):
  solo : herlev, riva, sipakmed (Metaplastic abnormal), sipakmed_metnormal (Metaplastic benign)
  mixed: herlev_sipakmed, herlev_sipakmed_metnormal, riva_herlev, sipakmed_riva, sipakmed_metnormal_riva

Normalization: solo runs use the source's own fold-k training statistics; mixed runs use ONE pooled fold-k training-only
statistic (both sources) for every image of the run (training, validation, holdout, unseen target).
Exports: every fit writes one prediction file with both OOD populations (fixed target holdout and entire target dataset).

Safe resumability (see training/ledger.py): persistent ``progress_manifest.json`` (pending/running/completed/failed), atomic
checkpoint / export / completion writes, integrity verification of completed fits on every start, quarantine (never delete) of
incomplete or corrupted outputs, exclusive campaign lock, identity check against the code + partition hashes, disk monitoring.

    uv run python train_v2.py --dry-run     # verify frozen files, enumerate the 720 fits, disk + identity preflight; trains nothing
    uv run python train_v2.py --status      # progress summary of an existing campaign
    uv run python train_v2.py               # full campaign; re-run the same command after any interruption to resume
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import datasets.protocol_v2 as P
from datasets.datasets import make_tf
from datasets.protocol_v2 import (
    EXPERIMENT_VERSION, MANIFEST_PATH, MIXED_RUNS, NUM_FOLDS, PRIMARY_MAPPING, SENSITIVITY_MAPPING, SOLO_RUNS,
    STATS_V2_PATH, load_mixed, load_partition, run_meta, verify_manifest,
)
from train_all_configs import ALL_MODEL_CONFIGS
from training import ledger as L
from training.export import export_fit_predictions
from training.io_utils import env_path, tee_log
from training.pipeline import train_dataset_v2, train_mixed_dataset_v2

REPO = Path(__file__).resolve().parent
SEED = 42
BALANCE_MODE = "weighted_loss"
BATCH_SIZE = 32
NUM_WORKERS = 8
USE_AMP = True
METRICS_DIR = env_path("METRICS_DIR_V3", "workspace", "metrics_v3_5fold")      # repository default; no override needed
RUNS_DIR = env_path("RUNS_DIR_V3", "workspace", "runs_v3_5fold")
EXPECTED_FITS = 720  # 9 runs x 16 models x 5 folds
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MIN_FREE_GB = 5.0          # stop cleanly below this
MARGIN_GB = 10.0           # preflight: remaining estimate + margin must fit
CKPT_MB = {"eat": 3, "iformer_m": 69, "efficientformerv2_s0": 26, "efficientformerv2_s1": 47, "efficientformerv2_s2": 98,
           "fastvit_t8": 26, "levit_128s": 56, "mobilenet_v4": 20, "mobilevitv2_100": 35, "efficientnet_b0": 32,
           "efficientnet_b1": 53, "efficientnet_b2": 62, "efficientnet_b3": 86, "mobilenetv2_100": 18,
           "tv_shufflenet_v2_x1_0": 10, "tv_squeezenet1_1": 6}      # mean best.pt size per model measured on the v1 checkpoints
PRED_MB = 6                                                       # generous per-fit allowance for the gz export + logs
CODE_FILES = ["train_v2.py", "train_all_configs.py", "train_models.py", "model_loader.py", "eat.py", "iformer.py", "pyproject.toml",
              "uv.lock", "datasets/datasets.py", "datasets/protocol_v2.py", "datasets/protocol_inputs/riva_slide_fold_assignment_5fold.csv",
              "training/pipeline.py", "training/engine.py", "training/export.py", "training/io_utils.py", "training/ledger.py",
              "training/eta.py", "data_analysis/active/lib/results_v2.py"]
GIT_GUARD_PATHS = CODE_FILES + ["datasets/partitions_v3_5fold", "training"]   # superseded v2 partitions are not training inputs


# ---------------------------------------------------------------------------------------------------------------------
# identity / reproducibility metadata
# ---------------------------------------------------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_state() -> dict:
    dirty = _git("status", "--porcelain", "--", *GIT_GUARD_PATHS)
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "dirty_tracked_code": bool(dirty),
            "dirty_entries": dirty.splitlines()[:20]}


def code_hashes() -> dict:
    return {f: L.sha256_file(REPO / f) for f in CODE_FILES}


def identity(matrix_size: int) -> dict:
    """Content identity of the experiment: any change of code, partitions or normalization statistics changes it."""
    return {"experiment_version": EXPERIMENT_VERSION, "n_fits": matrix_size, "code_sha256": hashlib.sha256(json.dumps(code_hashes(), sort_keys=True).encode()).hexdigest(),
            "partition_manifest_sha256": L.sha256_file(MANIFEST_PATH), "stats_sha256": L.sha256_file(STATS_V2_PATH)}


def environment() -> dict:
    deps = {d.metadata["Name"]: d.version for d in md.distributions() if d.metadata["Name"]}
    keys = ["torch", "torchvision", "timm", "scikit-learn", "numpy", "pandas", "scipy", "pillow", "tqdm"]
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "key_packages": {k: deps.get(k) for k in keys}, "all_packages": dict(sorted(deps.items(), key=lambda x: x[0].lower()))}


def transform_spec() -> dict:
    tr, ev = make_tf([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    return {"train": [repr(t) for t in tr.transforms], "eval": [repr(t) for t in ev.transforms]}


def fit_seed(key: str) -> int:
    """Per-fit seed: identical whether the fit runs in an uninterrupted campaign or after a resume."""
    return int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True    # v1 used benchmark=True / deterministic=False
    torch.backends.cudnn.benchmark = False


def run_matrix(runs=None, configs=None, folds=None) -> list[tuple[str, str, int]]:
    """(run_name, backbone_id, fold) for every fit."""
    runs = list(SOLO_RUNS) + [f"{a}_{b}" for a, b in MIXED_RUNS] if runs is None else list(runs)
    configs = ALL_MODEL_CONFIGS if configs is None else configs
    folds = range(NUM_FOLDS) if folds is None else folds
    return [(n, c.backbone_id, f) for n in runs for c in configs for f in folds]


def estimate_gb(matrix, keys=None) -> float:
    keys = set(keys) if keys is not None else None
    return sum((CKPT_MB.get(b, 100) * 1.25 + PRED_MB) / 1e3 for r, b, f in matrix if keys is None or L.fit_key(r, b, f) in keys)


def write_experiment_manifest(metrics_dir: Path, matrix, configs, allow_dirty: bool) -> dict:
    path = metrics_dir / "experiment_manifest.json"
    ident = identity(len(matrix)); g = git_state()
    if path.exists():
        man = json.loads(path.read_text())
        if man["identity"] != ident:
            raise L.IdentityMismatch(f"{path} was written under a different code/partition identity")
        man.setdefault("resumes", []).append({"at": L.now(), "git": g})
        L.atomic_write_json(path, man)
        return man
    stats_used = json.loads(STATS_V2_PATH.read_text())
    man = {"experiment_version": EXPERIMENT_VERSION, "created": L.now(), "identity": ident, "git": g, "allow_dirty": allow_dirty,
           "primary_endpoint": {"mapping": PRIMARY_MAPPING, "definition": "SIPaKMeD Metaplastic = normal/benign (runs *_metnormal)"},
           "sensitivity_endpoint": {"mapping": SENSITIVITY_MAPPING, "definition": "SIPaKMeD Metaplastic = abnormal (original mapping)"},
           "runs": {r: run_meta(r) for r in sorted({m[0] for m in matrix})}, "n_fits": len(matrix), "num_folds": NUM_FOLDS,
           "seeds": {"base_seed": SEED, "per_fit_seed": "sha256(f'{base_seed}|{run}|{backbone}|{fold}') -> int", "cudnn_deterministic": True, "cudnn_benchmark": False},
           "training": {"balance_mode": BALANCE_MODE, "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "amp": USE_AMP, "optimizer": "SGD (per model config)",
                        "scheduler": "MultiStepLR (per model config)", "loss": "CrossEntropyLoss(class weights from the training fold)",
                        "checkpoint_selection": "best validation accuracy"},
           "hyperparameters": {c.backbone_id: dataclasses.asdict(c) for c in configs},
           "transforms": transform_spec(),
           "normalization": {"policy": P.NORMALIZATION_POLICY, "stats_file": str(STATS_V2_PATH.relative_to(REPO)), "stats_sha256": ident["stats_sha256"],
                             "values": {k: v for k, v in stats_used.items() if k != "_meta"}},
           "partitions": json.loads(MANIFEST_PATH.read_text()), "code_sha256": code_hashes(), "environment": environment(),
           "evaluation_populations": {"primary": "eval_population == 'ood_target_holdout' (fixed holdout of the unseen target)",
                                      "supplementary": "all rows with eval_role == 'ood' (entire unseen target dataset; nested, no duplicated predictions)"},
           "resumes": []}
    L.atomic_write_json(path, man)
    return man


# ---------------------------------------------------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------------------------------------------------
def _purge_csv(path: Path, mask_fn) -> None:
    if path is None or not Path(path).exists() or Path(path).stat().st_size == 0:
        return
    df = pd.read_csv(path); drop = mask_fn(df)
    if not drop.any():
        return
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    df[~drop].to_csv(tmp, index=False); os.replace(tmp, path)


class Campaign:
    def __init__(self, metrics_dir: Path, matrix, configs, *, device, use_amp, ood_scope, min_free_gb, margin_gb, fault=None):
        self.dir = Path(metrics_dir); self.matrix = matrix; self.configs = configs; self.device = device; self.use_amp = use_amp
        self.ood_scope = ood_scope; self.min_free_gb = min_free_gb; self.margin_gb = margin_gb; self.fault = fault or (lambda stage, key: None)
        self.pred_dir = self.dir / "predictions"
        self.keys = [L.fit_key(*m) for m in matrix]
        self.led = L.FitLedger(self.dir / "progress_manifest.json", self.keys, identity(len(matrix)))
        self.session_failed: set[str] = set()
        self.cfg_by_backbone = {c.backbone_id: c for c in configs}

    def timing_csv(self, run: str) -> Path:
        """One timing table per run: solo and mixed pipelines write different column sets, so they must not share a file."""
        return self.dir / "training_time" / f"training_time_results_{run}.csv"

    def pred_path(self, run: str, backbone: str, fold: int) -> Path:
        return self.pred_dir / run / f"{backbone}__fold{fold}.csv.gz"

    def _rem_gb(self) -> float:
        pend = [k for k in self.keys if self.led.get(k)["status"] != "completed"]
        return estimate_gb(self.matrix, pend)

    # -- startup verification: every fit marked completed must still verify; otherwise it is reopened -------------------------
    def verify_completed(self) -> dict:
        out = {"verified": 0, "reopened": []}
        for key in self.led.keys_with("completed"):
            run, bb, fold = key.split("|"); fold = int(fold)
            fdir = self.dir / run / BALANCE_MODE / "campaign_v3" / self._slug(bb) / f"fold_{fold:02d}"
            ok, why = L.verify_fit(fdir, fdir / "checkpoints" / "best.pt", self.pred_path(run, bb, fold), key)
            if ok:
                out["verified"] += 1
            else:
                self.led.set(key, "failed", error=f"verification failed at start: {why}", finished=None)
                out["reopened"].append((key, why))
        return out

    def _slug(self, backbone: str) -> str:
        from training.io_utils import slugify
        return slugify(self.cfg_by_backbone[backbone].display_name)

    # -- hooks given to the pipeline --------------------------------------------------------------------------------------
    def guard(self, *, name, cfg, fold, fold_dir, best_ckpt_path, run_dir, epoch_file, summary_file) -> bool:
        key = L.fit_key(name, cfg.backbone_id, fold)
        if key in self.session_failed:
            return False
        pred = self.pred_path(name, cfg.backbone_id, fold)
        ok, why = L.verify_fit(fold_dir, best_ckpt_path, pred, key)
        if ok:                                                # verified complete: never retrain, never overwrite
            if self.led.get(key)["status"] != "completed":
                self.led.set(key, "completed", adopted_after_crash=True)
            return False
        if self.led.get(key)["status"] == "completed" or (fold_dir / "COMPLETE.json").exists():
            why_full = f"completed fit no longer verifies: {why}"
        else:
            why_full = "incomplete outputs from an interrupted fit" if L.has_content(fold_dir) or pred.exists() else ""
        if why_full:
            q = L.quarantine(self.dir, key, [fold_dir, pred], why_full)
            (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            print(f"[RESUME] {key}: {why_full} -> moved to {q}")
        _purge_csv(Path(epoch_file), lambda d: (d.dataset == name) & (d.model == cfg.display_name) & (d.fold == fold))
        _purge_csv(Path(summary_file), lambda d: (d.dataset == name) & (d.model == cfg.display_name) & (d.fold == fold))
        _purge_csv(self.timing_csv(name), lambda d: (d.dataset == name) & (d.backbone_id == cfg.backbone_id) & (d.fold == fold))
        info = L.check_disk(self.dir, remaining_gb=self._rem_gb(), min_free_gb=self.min_free_gb, margin_gb=self.margin_gb)
        self.led.d["disk"] = info
        if info.get("warning"):
            print(f"[DISK] WARNING {info['free_gb']} GB free < remaining estimate {info['remaining_estimate_gb']} GB + margin")
        self.led.set(key, "running", attempts=self.led.get(key)["attempts"] + 1, started=L.now(), error=None, finished=None)
        seed_everything(fit_seed(key))
        self.fault("guard", key)
        return True

    def make_export_cb(self, run_name: str):
        def _cb(*, fold, best_ckpt_path, cfg, display_name, backbone_id, origin):
            self.fault("export", L.fit_key(run_name, backbone_id, fold))
            out = export_fit_predictions(run_name=run_name, backbone_id=backbone_id, display_name=display_name, origin=origin, fold=fold,
                                         best_ckpt_path=best_ckpt_path, model_cfg=cfg, device=self.device, use_amp=self.use_amp,
                                         out_dir=self.pred_dir, batch_size=64, num_workers=NUM_WORKERS, ood_scope=self.ood_scope)
            print(f"[EXPORT] {out}")
        return _cb

    def done_cb(self, *, name, cfg, fold, fold_dir, best_ckpt_path) -> None:
        key = L.fit_key(name, cfg.backbone_id, fold)
        self.fault("done", key)
        pred = self.pred_path(name, cfg.backbone_id, fold)
        best_epoch = int(torch.load(best_ckpt_path, map_location="cpu", weights_only=False)["epoch"])
        rec = L.write_completion(fold_dir, best_ckpt_path, pred, key, {"best_epoch": best_epoch, "identity": self.led.d["identity"],
                                                                      "fit_seed": fit_seed(key), "git_commit": git_state()["commit"]})
        self.led.set(key, "completed", finished=L.now(), best_epoch=best_epoch, checkpoint_sha256=rec["checkpoint"]["sha256"],
                     predictions_sha256=rec["predictions"]["sha256"], prediction_rows=rec["predictions"]["rows"])
        c = self.led.counts()
        print(f"[PROGRESS] {c['completed']}/{len(self.keys)} completed | running {c['running']} | failed {c['failed']} | pending {c['pending']} | "
              f"free disk {L.free_gb(self.dir):.1f} GB")

    # -- main loop ------------------------------------------------------------------------------------------------------------
    def run(self, runs, folds, *, batch_size, num_workers, runs_dir, limit_rows=None) -> dict:
        v = self.verify_completed()
        print(f"[RESUME] verified completed fits: {v['verified']}; reopened: {len(v['reopened'])}")
        for k, why in v["reopened"]:
            print(f"[RESUME]   {k}: {why}")
        self.led.d["disk"] = L.check_disk(self.dir, remaining_gb=self._rem_gb(), min_free_gb=self.min_free_gb, margin_gb=self.margin_gb, preflight=True)
        self.led.save()
        try:
            for name in runs:
                keys = [L.fit_key(r, b, f) for r, b, f in self.matrix if r == name]
                while any(self.led.get(k)["status"] != "completed" and k not in self.session_failed for k in keys):
                    try:
                        self._train_run(name, folds, batch_size, num_workers, self.timing_csv(name), limit_rows)
                        break
                    except (KeyboardInterrupt, SystemExit, L.DiskSpaceError):
                        raise
                    except Exception as e:  # noqa: BLE001 - mark the fit failed, keep going with the rest
                        tb = traceback.format_exc(limit=3)
                        if not self.led.keys_with("running"):
                            raise                                  # failure outside any fit (e.g. data loading): do not loop
                        for k in self.led.keys_with("running"):
                            self.led.set(k, "failed", error=f"{type(e).__name__}: {e}", traceback=tb[-1500:], finished=L.now())
                            self.session_failed.add(k)
                            print(f"[FAILED] {k}: {type(e).__name__}: {e}")
                        gc.collect(); torch.cuda.empty_cache()
        finally:
            for k in self.led.keys_with("running"):        # interrupted (Ctrl-C, SIGTERM, disk stop): not failed, simply pending
                self.led.set(k, "pending", note=f"interrupted at {L.now()}")
        return self.led.counts()

    def _train_run(self, name, folds, batch_size, num_workers, results_csv, limit_rows) -> None:
        df = load_partition(name, verify_files=True) if name in SOLO_RUNS else None
        run_dir = self.dir / name / BALANCE_MODE / "campaign_v3"; run_dir.mkdir(parents=True, exist_ok=True)
        cfgs = [c for c in self.configs]
        common = dict(run_dir=run_dir, model_configs=cfgs, balance_mode=BALANCE_MODE, num_folds=NUM_FOLDS, batch_size=batch_size,
                      num_workers=num_workers, stats_path=STATS_V2_PATH, device=self.device, use_amp=self.use_amp, results_csv=results_csv,
                      export_cb=self.make_export_cb(name), fit_guard=self.guard, done_cb=self.done_cb, folds=folds)
        if name in SOLO_RUNS:
            train_dataset_v2(name=name, df=_limit(df, limit_rows), **common)
        else:
            a, b = [m for m in MIXED_RUNS if f"{m[0]}_{m[1]}" == name][0]
            train_mixed_dataset_v2(name=name, df=_limit(load_mixed(a, b), limit_rows), source_names=(a, b), pooled_norm_key=name, **common)


def _limit(df: pd.DataFrame, n):
    """Smoke tests only: keep n development rows per (source, fold)."""
    if not n:
        return df
    grp = ["source_dataset", "fold"] if "source_dataset" in df.columns else ["fold"]
    dev = df[df.split == "train_dev"]
    return pd.concat([dev.groupby(grp, group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=0)), df[df.split != "train_dev"]], ignore_index=True)


def run_campaign(*, metrics_dir=METRICS_DIR, runs_dir=RUNS_DIR, runs=None, configs=None, folds=None, device=DEVICE, use_amp=USE_AMP,
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, allow_dirty=False, ood_scope="full", min_free_gb=MIN_FREE_GB,
                 margin_gb=MARGIN_GB, limit_rows=None, fault=None, expected_fits=None, tee=True) -> dict:
    metrics_dir = Path(metrics_dir); runs_dir = Path(runs_dir)
    verify_manifest()                                              # abort if any frozen protocol file changed
    configs = ALL_MODEL_CONFIGS if configs is None else configs
    runs = list(SOLO_RUNS) + [f"{a}_{b}" for a, b in MIXED_RUNS] if runs is None else list(runs)
    matrix = run_matrix(runs, configs, folds)
    if expected_fits is not None:
        assert len(matrix) == expected_fits and len(set(matrix)) == len(matrix), len(matrix)
    g = git_state()
    if g["dirty_tracked_code"] and not allow_dirty:
        raise RuntimeError("tracked code/config has uncommitted changes (commit first, or pass --allow-dirty for tests): " + "; ".join(g["dirty_entries"][:5]))
    lock = L.CampaignLock(metrics_dir / ".campaign.lock"); lock.acquire()
    log_f = old_out = old_err = None
    try:
        metrics_dir.mkdir(parents=True, exist_ok=True); runs_dir.mkdir(parents=True, exist_ok=True)
        write_experiment_manifest(metrics_dir, matrix, configs, allow_dirty)
        if tee:
            log_f, old_out, old_err = tee_log(runs_dir / f"terminal_v3_{L.now().replace(' ', '_').replace(':', '-')}.log")
        camp = Campaign(metrics_dir, matrix, configs, device=device, use_amp=use_amp, ood_scope=ood_scope, min_free_gb=min_free_gb,
                        margin_gb=margin_gb, fault=fault)
        seed_everything(SEED)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        return camp.run(runs, folds, batch_size=batch_size, num_workers=num_workers, runs_dir=runs_dir, limit_rows=limit_rows)
    finally:
        if log_f is not None:
            sys.stdout, sys.stderr = old_out, old_err; log_f.close()
        lock.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="tests only: skip the clean-git-tree requirement")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args()
    verify_manifest()
    matrix = run_matrix()
    assert len(matrix) == EXPECTED_FITS and len(set(matrix)) == EXPECTED_FITS, len(matrix)
    print(f"[PROTOCOL v3] {len(matrix)} fits = {len(SOLO_RUNS) + len(MIXED_RUNS)} runs x {len(ALL_MODEL_CONFIGS)} models x {NUM_FOLDS} folds")
    led_path = METRICS_DIR / "progress_manifest.json"
    if args.status or args.dry_run:
        g = git_state(); need = estimate_gb(matrix)
        print(f"[OUTPUT] metrics={METRICS_DIR} runs={RUNS_DIR}")
        print(f"[DISK] free {L.free_gb(METRICS_DIR):.1f} GB | estimated need {need:.1f} GB (+{MARGIN_GB:.0f} GB margin) | minimum free to keep running {MIN_FREE_GB:.0f} GB")
        print(f"[GIT] commit {g['commit'][:10]} branch {g['branch']} dirty_tracked_code={g['dirty_tracked_code']}")
        print(f"[IDENTITY] {json.dumps(identity(len(matrix)))}")
        if led_path.exists():
            d = json.loads(led_path.read_text()); print(f"[LEDGER] {d.get('counts')} updated {d.get('updated')}")
        else:
            print("[LEDGER] none yet (fresh campaign)")
        if args.dry_run and L.free_gb(METRICS_DIR) < need + MARGIN_GB:
            sys.exit("[DISK] insufficient space for the full campaign")
        return
    counts = run_campaign(allow_dirty=args.allow_dirty, min_free_gb=args.min_free_gb, expected_fits=EXPECTED_FITS)
    print(f"[DONE] {counts}")
    sys.exit(0 if counts["completed"] == EXPECTED_FITS else 1)


if __name__ == "__main__":
    main()
