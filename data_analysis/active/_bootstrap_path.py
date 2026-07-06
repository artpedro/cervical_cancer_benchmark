"""Stdlib-only repo-root bootstrap for scripts run as ``python path/to/script.py``."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap(caller_file: str | Path) -> Path:
    for parent in Path(caller_file).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError(
        "Repository root not found (missing pyproject.toml). "
        "Run from the project tree or set PYTHONPATH to the repository root."
    )
