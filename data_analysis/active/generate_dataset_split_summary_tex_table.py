from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from datasets.datasets import scan_herlev, scan_riva, scan_sipakmed


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = Path("datasets/data")
OUTPUT_TEX = Path("workspace/analysis/dataset_split_summary_table.tex")

SEED = 42
NUM_FOLDS = 5
TEST_SIZE = 0.2

DATASET_ORDER = ("herlev", "sipakmed", "riva")
DATASET_DISPLAY = {"herlev": "Herlev", "sipakmed": "SIPaKMeD", "riva": "Riva"}


def _resolve_dataset_root(data_root: Path, dataset: str) -> Path:
    if dataset == "herlev":
        return data_root / "smear2005"
    if dataset == "sipakmed":
        return data_root / "sipakmed"
    if dataset == "riva":
        low = data_root / "riva"
        up = data_root / "RIVA"
        return low if low.exists() else up
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _scan(dataset: str, root: Path) -> pd.DataFrame:
    if dataset == "herlev":
        return scan_herlev(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    if dataset == "sipakmed":
        return scan_sipakmed(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    if dataset == "riva":
        return scan_riva(root=root, num_folds=NUM_FOLDS, seed=SEED, test_size=TEST_SIZE)
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _generate_table(counts: list[dict[str, int | str]]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[!tbph]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Dataset split summary with binary class counts after harmonization.}"
    )
    lines.append(r"  \label{tab:dataset_split_summary}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \hline")
    lines.append(
        r"    \textbf{Dataset} & \textbf{Split} & \textbf{Normal} & \textbf{Abnormal} & \textbf{Total} & \textbf{Abnormal (\%)} \\"
    )
    lines.append(r"    \hline")

    for r in counts:
        total = int(r["total"])
        abn = int(r["abnormal"])
        abn_pct = (100.0 * abn / total) if total > 0 else 0.0
        lines.append(
            f"    {r['dataset']} & {r['split']} & {int(r['normal'])} & {abn} & {total} & {abn_pct:.1f} \\\\"
        )
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    records: list[dict[str, int | str]] = []
    for ds in DATASET_ORDER:
        root = _resolve_dataset_root(DATA_ROOT.resolve(), ds)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found for {ds}: {root}")
        df = _scan(ds, root)

        for split in ("train_dev", "test"):
            sub = df[df["split"] == split]
            normal = int((sub["binary_label"] == "normal").sum())
            abnormal = int((sub["binary_label"] == "abnormal").sum())
            total = int(len(sub))
            records.append(
                {
                    "dataset": DATASET_DISPLAY[ds],
                    "split": split,
                    "normal": normal,
                    "abnormal": abnormal,
                    "total": total,
                }
            )

    tex = _generate_table(records)
    out = OUTPUT_TEX.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

