"""Repository root resolution for active analysis scripts."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not find repository root (missing pyproject.toml).")


REPO_ROOT = repo_root()
