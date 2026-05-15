from __future__ import annotations

from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
OUTPUT_TEX = Path("workspace/analysis/dataset_label_mapping_table.tex")

MAPPING: dict[str, dict[str, str]] = {
    "Herlev": {
        "normal_columnar": "Normal",
        "normal_intermediate": "Normal",
        "normal_superficiel": "Normal",
        "light_dysplastic": "Abnormal",
        "moderate_dysplastic": "Abnormal",
        "severe_dysplastic": "Abnormal",
        "carcinoma_in_situ": "Abnormal",
    },
    "SIPaKMeD": {
        "Superficial-Intermediate": "Normal",
        "Parabasal": "Normal",
        "Koilocytotic": "Abnormal",
        "Dyskeratotic": "Abnormal",
        "Metaplastic": "Abnormal",
    },
}

# Riva: (folder name in dataset, descriptive clinical name, binary mapping or None if excluded)
# "Not used" = present in RIVA taxonomy but excluded from the binary experiment (scan_riva).
RIVA_ROWS: list[tuple[str, str, str | None]] = [
    (
        "Sin_lesion",
        "NILM (Negative for Intraepithelial Lesion or Malignancy); same clinical role as no-lesion (Sin_lesion folder)",
        "Normal",
    ),
    ("ENDO", "ENDO (Endocervical Cells)", "Normal"),
    (
        "CA",
        "SCC (Squamous Cell Carcinoma); folder name in data is CA",
        "Abnormal",
    ),
    ("HSIL", "HSIL (High-Grade Squamous Intraepithelial Lesion)", "Abnormal"),
    (
        "ASC-H",
        "ASC-H (Atypical Squamous Cells --- Cannot Exclude High-Grade)",
        None,
    ),
    ("LSIL", "LSIL (Low-Grade Squamous Intraepithelial Lesion)", None),
    (
        "ASC-US",
        "ASC-US (Atypical Squamous Cells of Undetermined Significance)",
        None,
    ),
    ("INFL", "INFL (Inflammatory Cells)", None),
]


def _esc(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _mapping_cell(m: str | None) -> str:
    if m is None:
        return r"\textit{Not used in binary setup}"
    return _esc(m)


def _generate_table() -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[!tbph]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Binary label harmonization: original classes mapped to Normal/Abnormal. "
        r"For Riva, rows marked \textit{Not used in binary setup} denote classes present in the taxonomy but excluded from training/evaluation in this work (only four folders are loaded by \texttt{scan\_riva}).}"
    )
    lines.append(r"  \label{tab:binary_label_mapping}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \begin{tabular}{llp{6.2cm}l}")
    lines.append(r"    \hline")
    lines.append(
        r"    \textbf{Dataset} & \textbf{Folder / code} & \textbf{Descriptive name} & \textbf{Mapped binary} \\"
    )
    lines.append(r"    \hline")

    for dataset in ("Herlev", "SIPaKMeD"):
        rows = list(MAPPING[dataset].items())
        for i, (src_cls, dst) in enumerate(rows):
            ds = dataset if i == 0 else ""
            lines.append(
                f"    {_esc(ds)} & {_esc(src_cls)} & --- & {_esc(dst)} \\\\"
            )
        lines.append(r"    \hline")

    for i, (folder, desc, mapped) in enumerate(RIVA_ROWS):
        ds = "Riva" if i == 0 else ""
        lines.append(
            f"    {_esc(ds)} & {_esc(folder)} & {_esc(desc)} & {_mapping_cell(mapped)} \\\\"
        )
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
