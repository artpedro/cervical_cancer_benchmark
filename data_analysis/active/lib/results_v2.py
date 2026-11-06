"""
Common results loader for protocol v2 (3 folds, 9 runs, two SIPaKMeD label mappings).

Reads the per-image prediction exports written by ``training/export.py`` (one csv.gz per fit) and builds
  * ``per_image_table(preds, arm)``  : predictions with the arm-specific ground truth (``y_true``), for sample/slide-level resampling
  * ``build_master(preds, arm, scope)`` : checkpoint x target table in the SAME column layout as the v1
    ``generalization_master.csv`` so RQ1-RQ4 consume it unchanged (only the CSV path constant differs).

Endpoint arms (explicit everywhere):
  primary     : SIPaKMeD Metaplastic = normal/benign (runs ``*_metnormal``); SIPaKMeD *targets* use ``y_true_primary_metnormal``
  sensitivity : SIPaKMeD Metaplastic = abnormal (runs ``sipakmed``, ``sipakmed_riva``, ``herlev_sipakmed``); targets use ``y_true_sensitivity_metabnormal``
  Runs without SIPaKMeD as a training source (herlev, riva, riva_herlev) are 'shared' and appear in both arms; only the labels of
  their SIPaKMeD target rows differ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARM_LABEL = {"primary": "y_true_primary_metnormal", "sensitivity": "y_true_sensitivity_metabnormal"}
REQUIRED_EXPORT_COLUMNS = [
    "experiment_version", "run_name", "regime", "train_sources", "label_mapping", "endpoint_arm", "checkpoint_id",
    "checkpoint_sha256", "best_epoch", "backbone_id", "model", "origin", "fold", "target_dataset", "eval_role", "eval_population", "in_target_holdout", "split", "sample_id",
    "group_id", "label_full", "y_true_primary_metnormal", "y_true_sensitivity_metabnormal", "logit_normal",
    "logit_abnormal", "p_abnormal",
]
EPS = 1e-9


def load_predictions(pred_root: str | Path, *, expected_fits: int | None = None) -> pd.DataFrame:
    files = sorted(Path(pred_root).glob("*/*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no prediction exports under {pred_root}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    miss = [c for c in REQUIRED_EXPORT_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"export schema missing columns: {miss}")
    n_fits = df.checkpoint_id.nunique()
    if expected_fits is not None and n_fits != expected_fits:
        raise ValueError(f"expected {expected_fits} fits, found {n_fits}")
    per_ckpt = df.groupby("checkpoint_id")[["best_epoch", "checkpoint_sha256", "run_name", "fold"]].nunique()
    if (per_ckpt > 1).any().any():
        raise ValueError("checkpoint-level fields (best_epoch/sha256/run/fold) vary within a checkpoint")
    if df.experiment_version.nunique() != 1:
        raise ValueError("mixed experiment versions in prediction exports")
    return df


def select_arm(preds: pd.DataFrame, arm: str) -> pd.DataFrame:
    if arm not in ARM_LABEL:
        raise ValueError(arm)
    sub = preds[preds.endpoint_arm.isin([arm, "shared"])].copy()
    sub["endpoint_arm_analysis"] = arm
    sub["y_true"] = sub[ARM_LABEL[arm]].astype(int)
    sub["y_pred"] = (sub.p_abnormal > 0.5).astype(int)
    return sub


def per_image_table(preds: pd.DataFrame, arm: str) -> pd.DataFrame:
    return select_arm(preds, arm)


def _conf(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp + EPS); rec = tp / (tp + fn + EPS); spec = tn / (tn + fp + EPS)
    return dict(n=len(y), acc=(tp + tn) / (len(y) + EPS), prec=prec, rec=rec, spec=spec,
                f1=2 * prec * rec / (prec + rec + EPS), ppv=prec, npv=tn / (tn + fn + EPS))


def build_master(preds: pd.DataFrame, arm: str, scope: str = "test_split") -> pd.DataFrame:
    """scope: 'test_split' (OOD target restricted to its fixed holdout) or 'full_dataset' (all OOD target images)."""
    d = select_arm(preds, arm)
    # in-domain: holdout of each training-source dataset
    ind = d[d.eval_role == "test"]
    ind_rows = []
    for (cid, tgt), g in ind.groupby(["checkpoint_id", "target_dataset"], sort=False):
        c = _conf(g.y_true.to_numpy(), g.y_pred.to_numpy()); ind_rows.append(dict(checkpoint_id=cid, tgt=tgt, f1=c["f1"], acc=c["acc"]))
    ind_t = pd.DataFrame(ind_rows).groupby("checkpoint_id").agg(
        in_domain_f1_mean=("f1", "mean"), in_domain_f1_min=("f1", "min"), in_domain_acc_mean=("acc", "mean"),
        in_domain_acc_min=("acc", "min"), in_domain_n_eval_datasets=("tgt", "nunique"),
        in_domain_eval_datasets=("tgt", lambda s: ",".join(sorted(s)))).reset_index()
    ood = d[d.eval_role == "ood"]
    if scope == "test_split":
        ood = ood[ood.eval_population == "ood_target_holdout"]
    rows = []
    for (cid, tgt), g in ood.groupby(["checkpoint_id", "target_dataset"], sort=False):
        f = g.iloc[0]
        c = _conf(g.y_true.to_numpy(), g.y_pred.to_numpy())
        comps = f.train_sources.split(",")
        base = [("sipakmed" if s == "sipakmed_metnormal" else s) for s in comps]
        rows.append(dict(
            cfg_id=cid, source_dataset="_".join(base), source_dataset_regime=f.regime, source_dataset_components=",".join(base),
            source_stats_dataset="_".join(base), source_model=f.model, source_origin=f.origin, source_fold=int(f.fold),
            best_epoch=int(f.best_epoch), checkpoint_path=f.checkpoint_sha256, target_dataset=tgt, target_scope=scope,
            n_target_samples=c["n"], target_loss=np.nan, target_acc=c["acc"], target_prec=c["prec"], target_rec=c["rec"],
            target_spec=c["spec"], target_f1=c["f1"], target_ppv=c["ppv"], target_npv=c["npv"], target_seconds=np.nan,
            n_source_components=len(comps), is_in_domain_target=False,
            run_name=f.run_name, endpoint_arm=arm, label_mapping=f.label_mapping, run_endpoint_arm=f.endpoint_arm,
            experiment_version=f.experiment_version))
    m = pd.DataFrame(rows).merge(ind_t.rename(columns={"checkpoint_id": "cfg_id"}), on="cfg_id", how="left")
    m["gap_f1_to_target"] = m.in_domain_f1_mean - m.target_f1
    m["source_label"] = m.source_dataset + " -> " + m.source_model
    return m


def write_masters(pred_root: str | Path, out_dir: str | Path, *, scopes=("test_split", "full_dataset")) -> list[Path]:
    preds = load_predictions(pred_root)
    out = []
    for arm in ARM_LABEL:
        for sc in scopes:
            p = Path(out_dir) / f"generalization_master_v2__{arm}__{sc}.csv"
            build_master(preds, arm, sc).to_csv(p, index=False); out.append(p)
    return out
