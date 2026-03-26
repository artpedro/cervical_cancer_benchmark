from __future__ import annotations

import datetime
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


def env_path(env_name: str, *default_parts: str) -> Path:
    raw = os.getenv(env_name)
    if raw:
        return Path(raw).expanduser()
    return Path(*default_parts)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def append_csv_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    df = pd.DataFrame(rows).reindex(columns=columns)
    df.to_csv(path, mode="a", header=not file_exists, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def setup_fold_dir(run_dir: Path, model_name: str, fold: int) -> Path:
    fold_dir = run_dir / slugify(model_name) / f"fold_{fold:02d}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return fold_dir


def setup_run_dir(metrics_dir: Path, dataset_name: str, balance_mode: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = metrics_dir / dataset_name / balance_mode / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def tee_log(log_path: Path):
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")

    class _Tee:
        def __init__(self, *streams, isatty_stream=None):
            self.streams = streams
            self._isatty_stream = isatty_stream

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

        def isatty(self):
            if self._isatty_stream is None:
                return False
            return bool(getattr(self._isatty_stream, "isatty", lambda: False)())

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, log_f, isatty_stream=old_stdout)
    sys.stderr = _Tee(old_stderr, log_f, isatty_stream=old_stderr)
    return log_f, old_stdout, old_stderr

