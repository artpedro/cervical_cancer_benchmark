from __future__ import annotations

from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
OUTPUT_TEX = Path("workspace/analysis/augmentations_table.tex")


def _generate_table() -> str:
    # Mirrors datasets.datasets.make_tf exactly.
    train_rows = [
        ("1", "Resize", "size=256", "Always"),
        ("2", "CenterCrop", "size=224", "Always"),
        ("3", "RandomRotation", "degrees=180, interpolation=BILINEAR, fill=(255,255,255)", "Always"),
        ("4", "RandomHorizontalFlip", "p=0.5", "Bernoulli"),
        ("5", "ColorJitter", "brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0", "Always"),
        ("6", "ToTensor", "-", "Always"),
        ("7", "Normalize", "mean/std from normalization_stats.json (fallback ImageNet)", "Always"),
    ]
    eval_rows = [
        ("1", "Resize", "size=256", "Always"),
        ("2", "CenterCrop", "size=224", "Always"),
        ("3", "ToTensor", "-", "Always"),
        ("4", "Normalize", "mean/std from normalization_stats.json (fallback ImageNet)", "Always"),
    ]

    lines: list[str] = []
    lines.append(r"\begin{table*}[!tbph]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Image preprocessing and augmentation pipeline used in this work. "
        r"The training pipeline includes stochastic augmentations; the evaluation pipeline is deterministic.}"
    )
    lines.append(r"  \label{tab:augmentations}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \begin{tabular}{lllp{6.6cm}l}")
    lines.append(r"    \hline")
    lines.append(r"    \textbf{Pipeline} & \textbf{Step} & \textbf{Transform} & \textbf{Parameters} & \textbf{Applied} \\")
    lines.append(r"    \hline")

    for i, (step, name, params, applied) in enumerate(train_rows):
        pipe = "Train" if i == 0 else ""
        lines.append(f"    {pipe} & {step} & {name} & {params} & {applied} \\\\")
    lines.append(r"    \hline")
    for i, (step, name, params, applied) in enumerate(eval_rows):
        pipe = "Eval/Test" if i == 0 else ""
        lines.append(f"    {pipe} & {step} & {name} & {params} & {applied} \\\\")
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    tex = _generate_table()
    out = OUTPUT_TEX.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

