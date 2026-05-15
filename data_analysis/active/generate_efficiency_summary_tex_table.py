from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_analysis.active.dataset_regime_utils import is_mixed_dataset

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = Path("workspace/analysis/efficiency_profile/per_checkpoint_efficiency.csv")
OUTPUT_TEX = Path("workspace/analysis/efficiency_profile/efficiency_summary_table.tex")

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


def _fmt_pm(mean: float, std: float, decimals: int = 2) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _fmt_scalar(v: float, decimals: int = 2) -> str:
    return f"${v:.{decimals}f}$"


def _load() -> pd.DataFrame:
    if not INPUT_CSV.resolve().exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")
    df = pd.read_csv(INPUT_CSV.resolve(), low_memory=False)
    required = {
        "model",
        "dataset",
        "params_count",
        "macs_count",
        "flops_count",
        "latency_mean_ms_last_k",
        "memory_peak_mb_max_10_batches",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Efficiency CSV missing required columns: {missing}")
    if "latency_ms_basis" not in df.columns:
        df["latency_ms_basis"] = "batch"
    return df


def _generate_table(df: pd.DataFrame) -> str:
    include_mixed = bool(df["dataset"].astype(str).map(is_mixed_dataset).any())
    regime_label = "solo and mixed" if include_mixed else "solo-only"
    basis_mode = (
        df["latency_ms_basis"].dropna().astype(str).str.lower().mode()
    )
    basis = str(basis_mode.iloc[0]) if len(basis_mode) else "batch"
    latency_unit = "ms/image" if basis == "image" else "ms/batch"

    agg = (
        df.groupby("model", as_index=False)
        .agg(
            params_m=("params_count", lambda s: float(s.mean()) / 1e6),
            macs_g=("macs_count", lambda s: float(s.mean()) / 1e9),
            flops_g=("flops_count", lambda s: float(s.mean()) / 1e9),
            latency_mean=("latency_mean_ms_last_k", "mean"),
            latency_std=("latency_mean_ms_last_k", "std"),
            mem_peak_mean=("memory_peak_mb_max_10_batches", "mean"),
            mem_peak_std=("memory_peak_mb_max_10_batches", "std"),
        )
        .fillna(0.0)
    )

    lines: list[str] = []
    lines.append(r"\begin{table*}[!tbph]")
    lines.append(r"  \centering")
    lines.append(
        rf"  \caption{{Inference efficiency summary per model across all evaluation datasets/folds ({regime_label}). "
        rf"Latency reported as {latency_unit} (mean $\pm$ std); memory is peak allocated MB (mean $\pm$ std).}}"
    )
    lines.append(r"  \label{tab:efficiency_summary}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \hline")
    lines.append(
        rf"    \textbf{{Model}} & \textbf{{Params (M)}} & \textbf{{MACs (G)}} & "
        rf"\textbf{{FLOPs (G)}} & \textbf{{Latency ({latency_unit})}} & \textbf{{Peak Mem (MB)}} \\"
    )
    lines.append(r"    \hline")

    present = set(agg["model"].tolist())
    ordered = [m for m in MODEL_ORDER if m in present] + [
        m for m in sorted(present) if m not in MODEL_ORDER
    ]

    for model in ordered:
        row = agg.loc[agg["model"] == model].iloc[0]
        name = MODEL_TEX_NAMES.get(model, model.replace("_", r"\_"))
        lines.append(
            "    "
            + f"{name} & "
            + f"{_fmt_scalar(float(row['params_m']), 2)} & "
            + f"{_fmt_scalar(float(row['macs_g']), 3)} & "
            + f"{_fmt_scalar(float(row['flops_g']), 3)} & "
            + f"{_fmt_pm(float(row['latency_mean']), float(row['latency_std']), 3)} & "
            + f"{_fmt_pm(float(row['mem_peak_mean']), float(row['mem_peak_std']), 2)} "
            + r"\\"
        )

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = _load()
    tex = _generate_table(df)
    out = OUTPUT_TEX.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

