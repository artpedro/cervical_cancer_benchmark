from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/analysis/generalizability/plots/efficiency_tradeoff")
OUT_CSV = Path("workspace/analysis/generalizability/efficiency_generalization_summary.csv")


def _checkpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(
            ["cfg_id", "source_model", "source_origin", "source_fold", "best_epoch", "source_dataset_regime"],
            as_index=False,
        )
        .agg(
            mean_cross_f1=("target_f1", "mean"),
            worst_target_f1=("target_f1", "min"),
            latency_mean_ms=("latency_mean_ms", "mean"),
            params_count_mean=("params_count_mean", "mean"),
            macs_count_mean=("macs_count_mean", "mean"),
        )
    )
    return out


def _scatter(sub: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path) -> None:
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for regime, color in [("solo", "#1f77b4"), ("mixed", "#d62728")]:
        d = sub[sub["source_dataset_regime"].astype(str) == regime]
        if d.empty:
            continue
        ax.scatter(
            pd.to_numeric(d[x_col], errors="coerce"),
            pd.to_numeric(d[y_col], errors="coerce"),
            s=30,
            alpha=0.75,
            label=regime,
            c=color,
        )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if x_col in {"params_count_mean", "macs_count_mean"}:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    master_path = (MASTER_CSV if MASTER_CSV.is_absolute() else _REPO_ROOT / MASTER_CSV).resolve()
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master CSV: {master_path}")
    df = pd.read_csv(master_path, low_memory=False)
    required = {
        "cfg_id",
        "source_model",
        "source_origin",
        "source_fold",
        "best_epoch",
        "source_dataset_regime",
        "target_f1",
        "latency_mean_ms",
        "params_count_mean",
        "macs_count_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")

    df["target_f1"] = pd.to_numeric(df["target_f1"], errors="coerce")
    df = df.dropna(subset=["target_f1"]).copy()
    ckpt = _checkpoint_table(df)

    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for x_col in ("latency_mean_ms", "params_count_mean", "macs_count_mean"):
        _scatter(
            ckpt,
            x_col=x_col,
            y_col="mean_cross_f1",
            title=f"{x_col} vs mean_cross_f1 (solo vs mixed)",
            out_path=out_dir / f"{x_col}_vs_mean_cross_f1.png",
        )
        _scatter(
            ckpt,
            x_col=x_col,
            y_col="worst_target_f1",
            title=f"{x_col} vs worst_target_f1 (solo vs mixed)",
            out_path=out_dir / f"{x_col}_vs_worst_target_f1.png",
        )

    out_csv = (OUT_CSV if OUT_CSV.is_absolute() else _REPO_ROOT / OUT_CSV).resolve()
    ckpt.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
