"""
Revised experimental protocol (v3, 5-fold): persisted partitions for Herlev, SIPaKMeD and RIVA.

Supersedes the 3-fold v2 protocol (``datasets/partitions_v2/``, kept, marked SUPERSEDED).
Frozen design decisions (audit/pre_retraining/cv5/CV5_PROTOCOL_REPORT.md):
  * 5 development folds for every dataset; mixed runs pair fold i of each source (each source keeps its own grouping).
  * RIVA: the 5,793-crop dataset and the verified v1 holdout (``scan_riva``, seed 42) are unchanged. Development folds
    are whole inferred slides (``g_slide`` = category + slide number) assigned by the pre-registered constrained
    group-level balance search (``audit/pre_retraining/scripts/100_riva_5fold_search.py``); the frozen slide->fold
    table is ``datasets/protocol_inputs/riva_slide_fold_assignment_5fold.csv``.
  * SIPaKMeD: verified v1 holdout unchanged; ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` grouped by
    ``cluster_id`` and stratified on the ORIGINAL 5-class label -> identical fold membership for both label mappings.
  * Herlev: deterministic file order (sorted, case-sensitive ``.BMP``), holdout 20 % (seed 42),
    image-level ``StratifiedKFold(5, shuffle=True, random_state=42)`` (no group identifier exists).
  * Normalization statistics: each fold's TRAINING rows only (``fold != k``), Resize(224)+CenterCrop(224).

Only this module writes under ``datasets/partitions_v3_5fold/``; it never touches raw data, v1 stats or checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from datasets.datasets import compute_mean_std_for_df, scan_riva, scan_sipakmed

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "datasets" / "data"
PART_DIR = REPO / "datasets" / "partitions_v3_5fold"
STATS_PATH = PART_DIR / "normalization_stats_v3.json"
STATS_V2_PATH = STATS_PATH          # backwards-compatible alias used by train_v2 / export / validators
MANIFEST_PATH = PART_DIR / "MANIFEST.json"
SUPERSEDED_PART_DIRS = (REPO / "datasets" / "partitions_v2",)   # 3-fold protocol, kept for provenance
RIVA_ASSIGNMENT = REPO / "datasets" / "protocol_inputs" / "riva_slide_fold_assignment_5fold.csv"

NUM_FOLDS = 5
CV_SEED = 42         # Herlev / SIPaKMeD development-fold seed (repo default); RIVA uses the frozen slide assignment
HOLDOUT_SEED = 42    # holdout seed (unchanged from v1)
TEST_SIZE = 0.2

ROOTS = {"herlev": DATA_ROOT / "smear2005", "sipakmed": DATA_ROOT / "sipakmed", "riva": DATA_ROOT / "riva"}

HERLEV_NORMAL = {"normal_columnar", "normal_intermediate", "normal_superficiel"}
HERLEV_ABNORMAL = {"light_dysplastic", "moderate_dysplastic", "severe_dysplastic", "carcinoma_in_situ"}
SIP_NORMAL_A = {"Superficial-Intermediate", "Parabasal"}          # mapping A (as in v1): Metaplastic abnormal
SIP_ABNORMAL_A = {"Koilocytotic", "Dyskeratotic", "Metaplastic"}
SIP_NORMAL_B = SIP_NORMAL_A | {"Metaplastic"}                      # mapping B: Metaplastic benign
SIP_ABNORMAL_B = {"Koilocytotic", "Dyskeratotic"}
RIVA_NORMAL = {"Sin_lesion"}
RIVA_ABNORMAL = {"CA", "HSIL", "LSIL"}

# run-name -> (partition dataset, label mapping column)
DATASET_VARIANTS = {
    "herlev": ("herlev", "binary_idx"),
    "riva": ("riva", "binary_idx"),
    "sipakmed": ("sipakmed", "binary_idx"),
    "sipakmed_metnormal": ("sipakmed", "binary_idx_metnormal"),
}
SOLO_RUNS = ("herlev", "riva", "sipakmed", "sipakmed_metnormal")
MIXED_RUNS = (("herlev", "sipakmed"), ("herlev", "sipakmed_metnormal"), ("riva", "herlev"),
              ("sipakmed", "riva"), ("sipakmed_metnormal", "riva"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_riva_key(crop_path: str) -> tuple[str, str, str]:
    """crop <category...>_<slide>_<patch>_<A#>_<idx>_<label>.png -> (patch key, slide key, category).
    The label (class folder name, may contain '_' e.g. Sin_lesion) is removed first."""
    p = Path(crop_path)
    stem = p.stem.removesuffix("_" + p.parent.name)
    patch_tokens = stem.split("_")[:-2]           # drop annotator and running index
    category = "_".join(patch_tokens[:-2])
    slide = patch_tokens[-2]
    return "_".join(patch_tokens), f"{category}_{slide}", category


def _finish(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    root = ROOTS[dataset]
    df = df.copy()
    df["relpath"] = [str(Path(p).relative_to(root)) for p in df["path"]]
    df["sha256"] = [sha256_file(Path(p)) for p in df["path"]]
    df["sample_id"] = [f"{dataset}:{r}" for r in df["relpath"]]
    return df.drop(columns=["path"])


def build_herlev() -> pd.DataFrame:
    root = ROOTS["herlev"]
    recs = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        assert cls.name in HERLEV_NORMAL | HERLEV_ABNORMAL, cls.name
        for p in sorted(cls.glob("**/*"), key=lambda q: q.as_posix()):
            if p.is_file() and p.suffix == ".BMP":          # case-sensitive: excludes the *-d.bmp masks
                recs.append((str(p), cls.name))
    assert len(recs) == 917, len(recs)
    df = pd.DataFrame(recs, columns=["path", "label_full"])
    df["binary_idx"] = df.label_full.isin(HERLEV_ABNORMAL).astype(int)
    df["binary_idx_metnormal"] = df["binary_idx"]
    df["group_id"] = df["path"].map(lambda p: Path(p).stem)   # no group identifier exists: image-level
    tr, te = train_test_split(df.index, test_size=TEST_SIZE, random_state=HOLDOUT_SEED, stratify=df.binary_idx)
    df["split"] = "train_dev"; df.loc[te, "split"] = "test"
    df["fold"] = -1
    dev = df[df.split == "train_dev"]
    for k, (_, v) in enumerate(StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=CV_SEED).split(dev, dev.binary_idx)):
        df.loc[dev.index[v], "fold"] = k
    return _finish(df, "herlev")


def build_sipakmed() -> pd.DataFrame:
    df = scan_sipakmed(ROOTS["sipakmed"], num_folds=5, seed=HOLDOUT_SEED, test_size=TEST_SIZE)   # holdout: unchanged v1 split
    df = df.drop(columns=["fold"])
    df["binary_idx"] = df.label_full.isin(SIP_ABNORMAL_A).astype(int)
    df["binary_idx_metnormal"] = df.label_full.isin(SIP_ABNORMAL_B).astype(int)
    df["group_id"] = df["cluster_id"]
    assert (df.groupby("cluster_id").label_full.nunique() == 1).all()
    df["fold"] = -1
    dev = df[df.split == "train_dev"]
    sg = StratifiedGroupKFold(NUM_FOLDS, shuffle=True, random_state=CV_SEED)
    for k, (_, v) in enumerate(sg.split(dev, dev.label_full, dev.cluster_id)):      # stratify on the 5-class label
        df.loc[dev.index[v], "fold"] = k
    return _finish(df, "sipakmed")


def build_riva() -> pd.DataFrame:
    df = scan_riva(ROOTS["riva"], num_folds=5, seed=HOLDOUT_SEED, test_size=TEST_SIZE)           # holdout: unchanged v1 split
    df = df.drop(columns=["fold"]).rename(columns={"cluster_id": "g_repo_v1"})
    keys = df["path"].map(parse_riva_key)
    df["g_image"] = [k[0] for k in keys]; df["g_slide"] = [k[1] for k in keys]; df["category"] = [k[2] for k in keys]
    df["binary_idx_metnormal"] = df["binary_idx"]
    df["group_id"] = df["g_slide"]
    asg = pd.read_csv(RIVA_ASSIGNMENT)
    assert asg.slide.is_unique and set(asg.fold) == set(range(NUM_FOLDS))
    dev_slides = set(df.loc[df.split == "train_dev", "g_slide"])
    assert set(asg.slide) == dev_slides, "slide assignment does not cover exactly the development slides"
    assert not (set(asg.slide) & set(df.loc[df.split == "test", "g_slide"])), "holdout slide in fold assignment"
    m = dict(zip(asg.slide, asg.fold))
    df["fold"] = [m[g] if sp == "train_dev" else -1 for g, sp in zip(df.g_slide, df.split)]
    return _finish(df, "riva")


def compute_stats_v2(parts: dict[str, pd.DataFrame]) -> dict:
    stats: dict = {"_meta": {"method": "mean/std over each fold's TRAINING rows only (fold != k, split == train_dev); "
                                       "Resize(224)+CenterCrop(224)+ToTensor; float64 accumulation",
                             "num_folds": NUM_FOLDS, "protocol": "protocol_v3_5fold"}}
    for ds, df in parts.items():
        dev = df[df.split == "train_dev"]
        stats[ds] = {}
        for k in range(NUM_FOLDS):
            tr = dev[dev.fold != k].copy()
            tr["path"] = [str(ROOTS[ds] / r) for r in tr.relpath]
            m, s = compute_mean_std_for_df(tr.reset_index(drop=True), resize=224, crop_size=224, show_progress=False)
            stats[ds][f"fold_{k}"] = {"mean": m, "std": s, "n_train": int(len(tr))}
    stats["sipakmed_metnormal"] = stats["sipakmed"]     # identical images/folds -> identical statistics (alias key)
    add_mixed_pooled_stats(stats)
    return stats


NORMALIZATION_POLICY = {"solo": "the source's own fold-k training statistics for all images of the run",
                        "mixed": "pooled fold-k training statistics of the two sources (training rows only) for ALL images of the run, "
                                 "including validation, holdout and unseen-target images"}


def add_mixed_pooled_stats(stats: dict) -> dict:
    """Mixed-run normalization (revision): one pooled mean/std per mixed run and fold, computed exclusively from the TRAINING
    rows of the two sources (fold != k, split == train_dev). Every image of a mixed run (training, validation, holdout, unseen
    target) is normalised with these values. Exact pooling of population statistics: all images are 224x224 after the
    deterministic transform, so weights are the sources' n_train; mean = sum w m, var = sum w (s^2 + m^2) - mean^2."""
    for a, b in MIXED_RUNS:
        key = f"{a}_{b}"
        bases = [DATASET_VARIANTS[a][0], DATASET_VARIANTS[b][0]]
        stats[key] = {}
        for k in range(NUM_FOLDS):
            e = [stats[x][f"fold_{k}"] for x in bases]
            n = np.array([x["n_train"] for x in e], dtype=float); w = n / n.sum()
            m = np.array([x["mean"] for x in e]); sd = np.array([x["std"] for x in e])
            mean = (w[:, None] * m).sum(0); var = (w[:, None] * (sd ** 2 + m ** 2)).sum(0) - mean ** 2
            stats[key][f"fold_{k}"] = {"mean": mean.tolist(), "std": np.sqrt(var).tolist(), "n_train": int(n.sum()),
                                       "sources": bases, "n_train_by_source": [int(x) for x in n]}
    return stats


def norm_key(run_name: str) -> str:
    """Key of the normalization statistics used for EVERY image of a run (solo: the source's own; mixed: the pooled entry)."""
    run_sources(run_name)   # validates the name
    return run_name


def build_all(write: bool = True) -> dict[str, pd.DataFrame]:
    parts = {"herlev": build_herlev(), "sipakmed": build_sipakmed(), "riva": build_riva()}
    if write:
        PART_DIR.mkdir(parents=True, exist_ok=True)
        for ds, df in parts.items():
            df.to_csv(PART_DIR / f"{ds}_partition.csv", index=False)
        stats = compute_stats_v2(parts)
        STATS_PATH.write_text(json.dumps(stats, indent=2))
        import sklearn
        man = {"protocol": EXPERIMENT_VERSION, "num_folds": NUM_FOLDS, "cv_seed_herlev_sipakmed": CV_SEED,
               "riva_fold_source": str(RIVA_ASSIGNMENT.relative_to(REPO)), "riva_assignment_sha256": sha256_file(RIVA_ASSIGNMENT),
               "superseded": [str(p.relative_to(REPO)) for p in SUPERSEDED_PART_DIRS],
               "holdout_seed": HOLDOUT_SEED, "test_size": TEST_SIZE, "normalization": NORMALIZATION_POLICY,
               "sklearn": sklearn.__version__, "python": sys.version.split()[0],
               "files": {p.name: sha256_file(p) for p in sorted(PART_DIR.glob("*")) if p.name != "MANIFEST.json"}}
        MANIFEST_PATH.write_text(json.dumps(man, indent=2))
    return parts


def load_partition(run_name: str, *, verify_files: bool = False) -> pd.DataFrame:
    """Frozen partition in the layout the existing training pipeline expects.

    Columns: path (absolute), label_full, binary_idx (per label mapping of ``run_name``), binary_label, split, fold,
    cluster_id (group id), sample_id.  ``run_name`` in DATASET_VARIANTS.
    """
    ds, col = DATASET_VARIANTS[run_name]
    df = pd.read_csv(PART_DIR / f"{ds}_partition.csv")
    root = ROOTS[ds]
    df["path"] = [str(root / r) for r in df.relpath]
    if verify_files:
        bad = [r for r, h in zip(df.path, df.sha256) if sha256_file(Path(r)) != h]
        if bad:
            raise RuntimeError(f"{len(bad)} files differ from the frozen manifest, e.g. {bad[:2]}")
    df["binary_idx"] = df[col].astype(int)
    df["binary_label"] = np.where(df.binary_idx == 1, "abnormal", "normal")
    df["cluster_id"] = df["group_id"]
    return df


def load_mixed(name_a: str, name_b: str) -> pd.DataFrame:
    a = load_partition(name_a).assign(source_dataset=name_a)
    b = load_partition(name_b).assign(source_dataset=name_b)
    return pd.concat([a, b], ignore_index=True)


def verify_manifest() -> None:
    man = json.loads(MANIFEST_PATH.read_text())
    for name, h in man["files"].items():
        if sha256_file(PART_DIR / name) != h:
            raise RuntimeError(f"Frozen protocol file changed: {name}")
    if sha256_file(RIVA_ASSIGNMENT) != man["riva_assignment_sha256"]:
        raise RuntimeError("Frozen RIVA slide->fold assignment changed")




# ---------------------------------------------------------------------------------------------------------------------
# Experiment metadata: endpoint designation (PRIMARY = Metaplastic -> normal/benign; SENSITIVITY = Metaplastic -> abnormal)
# ---------------------------------------------------------------------------------------------------------------------
EXPERIMENT_VERSION = "protocol_v3_5fold"
PRIMARY_MAPPING = "metnormal"          # SIPaKMeD Metaplastic = normal/benign (revised primary endpoint)
SENSITIVITY_MAPPING = "metabnormal"    # SIPaKMeD Metaplastic = abnormal (original mapping, sensitivity condition)
ENDPOINT_ARMS = ("primary", "sensitivity")


def run_names() -> list[str]:
    return list(SOLO_RUNS) + [f"{a}_{b}" for a, b in MIXED_RUNS]


def run_sources(run_name: str) -> tuple[str, ...]:
    """Run-variant names of the training sources, e.g. 'sipakmed_metnormal_riva' -> ('sipakmed_metnormal', 'riva')."""
    if run_name in SOLO_RUNS:
        return (run_name,)
    for a, b in MIXED_RUNS:
        if run_name == f"{a}_{b}":
            return (a, b)
    raise KeyError(run_name)


def base_dataset(variant: str) -> str:
    return DATASET_VARIANTS[variant][0]


def run_meta(run_name: str) -> dict:
    """Explicit endpoint metadata of a run.

    endpoint_arm: 'primary' (trained with the metnormal mapping), 'sensitivity' (metabnormal mapping), or 'shared'
    (SIPaKMeD is not a training source: the run belongs to both arms; only the labels of SIPaKMeD *targets* differ by arm).
    """
    srcs = run_sources(run_name)
    bases = tuple(base_dataset(s) for s in srcs)
    if "sipakmed_metnormal" in srcs:
        mapping, arm = PRIMARY_MAPPING, "primary"
    elif "sipakmed" in srcs:
        mapping, arm = SENSITIVITY_MAPPING, "sensitivity"
    else:
        mapping, arm = "not_applicable", "shared"
    return dict(run_name=run_name, regime="solo" if len(srcs) == 1 else "mixed", train_sources=",".join(srcs),
                train_base_datasets=",".join(bases), source_dataset="_".join(bases), label_mapping=mapping,
                endpoint_arm=arm, experiment_version=EXPERIMENT_VERSION)


if __name__ == "__main__":
    build_all(write=True)
    print("wrote", PART_DIR)
