"""
Persistent progress ledger, completion manifests, integrity verification, quarantine and disk monitoring for the v3 campaign.

Guarantees
  * Every state change of every fit (pending / running / completed / failed) is persisted atomically (tmp file + fsync + rename).
  * A fit is *completed* only when ``COMPLETE.json`` exists in its fold directory AND the recorded sha256 of the checkpoint and
    of the prediction export still match the files, the checkpoint archive passes a CRC test and the prediction file is
    readable with the recorded row count. Completion is written last, after checkpoint and export are on disk.
  * Completed + verified fits are never touched. Incomplete or corrupted outputs are MOVED to ``_quarantine`` (never deleted)
    before the fit is retrained, so nothing is silently overwritten and nothing is lost.
  * One campaign per metrics directory (exclusive file lock); a ledger created under a different code / partition identity
    is refused instead of mixing results.
"""
from __future__ import annotations

import datetime
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

STATUSES = ("pending", "running", "completed", "failed")


class DiskSpaceError(RuntimeError):
    pass


class IdentityMismatch(RuntimeError):
    pass


class CampaignLocked(RuntimeError):
    pass


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def fit_key(run: str, backbone_id: str, fold: int) -> str:
    return f"{run}|{backbone_id}|{fold}"


class CampaignLock:
    """Exclusive non-blocking lock so two campaigns can never write into the same metrics directory."""

    def __init__(self, path: Path):
        self.path = Path(path); self.fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self.fh.close(); self.fh = None
            raise CampaignLocked(f"another campaign holds {self.path}") from e
        self.fh.seek(0); self.fh.truncate(); self.fh.write(f"pid={os.getpid()} since={now()}\n"); self.fh.flush()

    def release(self) -> None:
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN); self.fh.close(); self.fh = None


class FitLedger:
    def __init__(self, path: Path, keys: Iterable[str], identity: dict):
        self.path = Path(path); keys = list(keys)
        if self.path.exists():
            d = json.loads(self.path.read_text())
            if d["identity"] != identity:
                diff = {k: (d["identity"].get(k), identity.get(k)) for k in set(d["identity"]) | set(identity) if d["identity"].get(k) != identity.get(k)}
                raise IdentityMismatch(f"ledger {self.path} was created under a different identity: {diff}")
            if set(d["fits"]) != set(keys):
                raise IdentityMismatch("ledger fit set differs from the requested matrix")
            self.d = d
        else:
            self.d = {"schema": 1, "created": now(), "identity": identity, "disk": {},
                      "fits": {k: {"status": "pending", "attempts": 0} for k in keys}}
            self.save()

    def save(self) -> None:
        self.d["updated"] = now()
        self.d["counts"] = self.counts()
        atomic_write_json(self.path, self.d)

    def counts(self) -> dict:
        c = {s: 0 for s in STATUSES}
        for v in self.d["fits"].values():
            c[v["status"]] += 1
        return c

    def get(self, key: str) -> dict:
        return self.d["fits"][key]

    def set(self, key: str, status: str, **fields) -> None:
        assert status in STATUSES
        e = self.d["fits"][key]; e["status"] = status; e.update(fields)
        self.save()

    def keys_with(self, status: str) -> list[str]:
        return [k for k, v in self.d["fits"].items() if v["status"] == status]


# ---------------------------------------------------------------------------------------------------------------------
# completion manifest + verification
# ---------------------------------------------------------------------------------------------------------------------
def _gz_rows(p: Path) -> int:
    n = 0
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n - 1     # header


def write_completion(fold_dir: Path, ckpt: Path, pred: Path, key: str, extra: dict) -> dict:
    rec = {"key": key, "completed_at": now(),
           "checkpoint": {"file": ckpt.name, "bytes": ckpt.stat().st_size, "sha256": sha256_file(ckpt)},
           "predictions": {"file": str(pred), "bytes": pred.stat().st_size, "sha256": sha256_file(pred), "rows": _gz_rows(pred)}, **extra}
    atomic_write_json(Path(fold_dir) / "COMPLETE.json", rec)
    return rec


def verify_fit(fold_dir: Path, ckpt: Path, pred: Path, key: str) -> tuple[bool, str]:
    cj = Path(fold_dir) / "COMPLETE.json"
    if not cj.exists():
        return False, "no COMPLETE.json"
    try:
        rec = json.loads(cj.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable COMPLETE.json: {e}"
    if rec.get("key") != key:
        return False, "completion manifest belongs to another fit"
    if not ckpt.exists() or not pred.exists():
        return False, "checkpoint or prediction file missing"
    if ckpt.stat().st_size != rec["checkpoint"]["bytes"] or pred.stat().st_size != rec["predictions"]["bytes"]:
        return False, "size mismatch"
    if sha256_file(ckpt) != rec["checkpoint"]["sha256"]:
        return False, "checkpoint sha256 mismatch"
    try:
        with zipfile.ZipFile(ckpt) as z:
            bad = z.testzip()
        if bad:
            return False, f"checkpoint archive CRC failure in {bad}"
    except Exception as e:  # noqa: BLE001
        return False, f"checkpoint archive unreadable: {e}"
    if sha256_file(pred) != rec["predictions"]["sha256"]:
        return False, "prediction sha256 mismatch"
    try:
        if _gz_rows(pred) != rec["predictions"]["rows"]:
            return False, "prediction row count mismatch"
    except Exception as e:  # noqa: BLE001
        return False, f"prediction file unreadable: {e}"
    return True, "ok"


def quarantine(metrics_dir: Path, key: str, paths: Iterable[Path], reason: str) -> Path | None:
    """Move existing leftovers aside (never delete). Returns the quarantine folder or None if nothing was there."""
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return None
    qdir = Path(metrics_dir) / "_quarantine" / (key.replace("|", "__") + "__" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    qdir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(existing):
        shutil.move(str(p), str(qdir / f"{i}_{p.name}"))
    atomic_write_json(qdir / "REASON.json", {"key": key, "reason": reason, "moved": [str(p) for p in existing], "at": now()})
    return qdir


def has_content(fold_dir: Path) -> bool:
    return any(p.is_file() for p in Path(fold_dir).rglob("*"))


# ---------------------------------------------------------------------------------------------------------------------
# disk monitoring
# ---------------------------------------------------------------------------------------------------------------------
def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def check_disk(path: Path, *, remaining_gb: float, min_free_gb: float, margin_gb: float, preflight: bool = False) -> dict:
    f = free_gb(path)
    info = {"free_gb": round(f, 2), "remaining_estimate_gb": round(remaining_gb, 2), "checked_at": now()}
    need = remaining_gb + margin_gb
    if preflight and f < need:
        raise DiskSpaceError(f"preflight: {f:.1f} GB free < {need:.1f} GB needed ({remaining_gb:.1f} GB estimated + {margin_gb:.1f} GB margin)")
    if f < min_free_gb:
        raise DiskSpaceError(f"only {f:.1f} GB free (< {min_free_gb:.1f} GB minimum); stopping cleanly, restart resumes")
    info["warning"] = bool(f < need)
    return info
