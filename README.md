# Cervical-cancer CNN + Hybrid Transformers

Binary pap-smear classification (train CNNs and Hybrid Transformerson SIPaKMeD / Riva / Herlev, mixed runs, CV).

## Setup

```bash
uv sync
```

Requires Python ≥ 3.13 (see `pyproject.toml`).

## Data layout

Point **`DATA_ROOT`** at the folder that contains one subdirectory per dataset (default: `datasets/data`):

- `sipakmed/`, `riva/`, `herlev/` — each with the layout expected by the scanners in `datasets/datasets.py`.

If a dataset folder is missing, `train_all_configs.py` / `train_mixed_models.py` skip jobs that need it and print a warning.

## Run training

From the repo root, with the venv active or via `uv run`:

| Command | What it does |
|--------|----------------|
| `uv run python train_models.py` | Single-dataset training (sipakmed + riva in config). |
| `uv run python train_mixed_models.py` | Mixed pairs only (sipakmed–riva, riva–herlev, herlev–sipakmed when roots exist). |
| `uv run python train_all_configs.py` | Solo on all available datasets, then mixed pairs. |

Optional env vars (defaults in each script): `DATA_ROOT`, `METRICS_DIR` (default `workspace/metricsv2`), `RUNS_DIR` (default `workspace/runsv2`).

## Folders

| Path | Role |
|------|------|
| **`datasets/`** | Data loading: `datasets.py` (scans, folds, loaders), `normalization_stats.json`, optional `data/` for image roots. |
| **`training/`** | Training code: `pipeline.py` (train/eval loops), `engine.py`, `io_utils.py`, `eta.py`. |
| **`workspace/`** | Outputs: metrics runs, logs, analysis figures (exact subfolders vary by experiment; `metricsv2` / `runsv2` are defaults for current scripts). |
| **`data_analysis/`** | Offline analysis and plots: `active/` (current scripts), `legacy/` (older). |

Top-level: `model_loader.py` (backbone registry), `eat.py` / `iformer.py` (custom models), `test_modules.py` (smoke tests).
