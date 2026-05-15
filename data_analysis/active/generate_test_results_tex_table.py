from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from data_analysis.active.dataset_regime_utils import (
    dataset_display_name,
    infer_dataset_order,
    is_mixed_dataset,
)

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = Path("workspace/analysis/test_eval_results/per_checkpoint_test_metrics.csv")
OUTPUT_TEX = Path("workspace/analysis/test_eval_results/test_results_table.tex")

DATASET_ORDER: list[str] | None = None
INCLUDE_MIXED_DATASETS = True
DATASET_INCLUDE: set[str] | None = None

MODEL_ORDER = [
    "EfficientNet B0",
    "EfficientNet B1",
    "EfficientNet B2",
    "MobileNet V2",
    "MobileNet V4",
    "EfficientFormerV2 S0",
    "EfficientFormerV2 S1",
    "FastViT T8",
    "MobileViT v2 100",
    "iformer_m",
]

MODEL_TEX_NAMES: dict[str, str] = {
    "EfficientNet B0": "EfficientNet-B0",
    "EfficientNet B1": "EfficientNet-B1",
    "EfficientNet B2": "EfficientNet-B2",
    "MobileNet V2": "MobileNetV2",
    "MobileNet V4": "MobileNetV4",
    "EfficientFormerV2 S0": "EfficientFormerV2-S0",
    "EfficientFormerV2 S1": "EfficientFormerV2-S1",
    "FastViT T8": "FastViT-T8",
    "MobileViT v2 100": "MobileViT-V2",
    "iformer_m": "IFormer-M",
}

METRICS = [
    ("test_bal_acc", "Bal.\\ Acc."),
    ("test_f1", "F1-score"),
    ("test_rec", "Recall"),
    ("test_spec", "Specificity"),
    ("test_prec", "Precision"),
]

DECIMALS = 3


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "test_bal_acc" in df.columns:
        print("[INFO] Using test_bal_acc directly from CSV.")
    else:
        print("[WARN] test_bal_acc not found in CSV; deriving it from test_rec and test_spec.")
        df["test_bal_acc"] = (df["test_rec"] + df["test_spec"]) / 2.0
    return df


def _fmt(mean: float, std: float, bold: bool) -> str:
    s = f"{mean:.{DECIMALS}f} \\pm {std:.{DECIMALS}f}"
    if bold:
        return f"$\\mathbf{{{s}}}$"
    return f"${s}$"


def _generate_table(df: pd.DataFrame) -> str:
    metric_cols = [m[0] for m in METRICS]
    metric_headers = [m[1] for m in METRICS]

    agg = (
        df.groupby(["dataset", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    agg.columns = ["dataset", "model"] + [
        f"{m}_{stat}" for m in metric_cols for stat in ("mean", "std")
    ]

    datasets_present = sorted(df["dataset"].dropna().astype(str).unique().tolist())
    dataset_order = DATASET_ORDER or infer_dataset_order(datasets_present)
    if not INCLUDE_MIXED_DATASETS:
        dataset_order = [ds for ds in dataset_order if not is_mixed_dataset(ds)]
    if DATASET_INCLUDE is not None:
        dataset_order = [ds for ds in dataset_order if ds in DATASET_INCLUDE]
    if not dataset_order:
        raise RuntimeError("No datasets selected for table generation.")

    best: dict[tuple[str, str], float] = {}
    for ds in dataset_order:
        sub = agg[agg["dataset"] == ds]
        if sub.empty:
            continue
        for mc in metric_cols:
            best[(ds, mc)] = float(sub[f"{mc}_mean"].max())

    n_models = len(MODEL_ORDER)
    n_metrics = len(METRICS)
    col_spec = "ll" + "c" * n_metrics

    lines: list[str] = []
    lines.append(r"\begin{table*}[!tbph]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Test-set performance of the evaluated models by evaluation dataset. "
        r"Results are reported as mean $\pm$ standard deviation across folds. "
        r"Best value per dataset and metric is highlighted in bold.}"
    )
    lines.append(r"  \label{tab:test_results}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{2pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \hline")

    header = r"    \textbf{Dataset} & \textbf{Model}"
    for h in metric_headers:
        header += f" & \\textbf{{{h}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"    \hline")

    for di, ds in enumerate(dataset_order):
        ds_display = dataset_display_name(ds)
        sub = agg[agg["dataset"] == ds]

        models_present = [m for m in MODEL_ORDER if m in sub["model"].values]
        models_present += [m for m in sorted(sub["model"].astype(str).unique().tolist()) if m not in models_present]
        n_present = len(models_present)
        if n_present == 0:
            continue

        for mi, model in enumerate(models_present):
            row = sub[sub["model"] == model].iloc[0]
            tex_name = MODEL_TEX_NAMES.get(model, model)

            if mi == 0:
                prefix = f"    \\multirow{{{n_present}}}{{*}}{{\\shortstack[l]{{{ds_display}}}}}"
            else:
                prefix = "   "

            cells: list[str] = []
            for mc in metric_cols:
                mean_val = float(row[f"{mc}_mean"])
                std_val = float(row[f"{mc}_std"])
                is_best = np.isclose(mean_val, best.get((ds, mc), mean_val), atol=1e-6)
                cells.append(_fmt(mean_val, std_val, is_best))

            line = f"{prefix} & {tex_name:<24s} & " + " & ".join(cells) + r" \\"
            lines.append(line)

        lines.append(r"    \hline")

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


def main() -> None:
    csv_path = INPUT_CSV.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = _load(csv_path)
    tex = _generate_table(df)

    out = OUTPUT_TEX.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"\n[OK] wrote {out}")


if __name__ == "__main__":
    main()
