from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


MASTER_CSV = Path("workspace/analysis/generalizability/generalization_master.csv")
OUT_DIR = Path("workspace/analysis/generalizability")
OUT_CSV = OUT_DIR / "generalizability_summary.csv"  # backward-compatible summary
OUT_TEX = OUT_DIR / "generalizability_summary.tex"  # backward-compatible summary

DIFF_CSV = OUT_DIR / "dataset_difficulty_summary.csv"
DIFF_TEX = OUT_DIR / "dataset_difficulty_summary.tex"
MIXED_CSV = OUT_DIR / "best_models_by_mixed_source.csv"
MIXED_TEX = OUT_DIR / "best_models_by_mixed_source.tex"
REGIME_CSV = OUT_DIR / "regime_comparison_summary.csv"
REGIME_TEX = OUT_DIR / "regime_comparison_summary.tex"

CKPT_KEYS = [
    "cfg_id",
    "source_dataset",
    "source_dataset_regime",
    "source_model",
    "source_origin",
    "source_fold",
    "best_epoch",
]


def _tex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def _pretty_dataset(name: str) -> str:
    return str(name).replace("_", "+")


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _is_in_domain_series(df: pd.DataFrame) -> pd.Series:
    if "is_in_domain_target" in df.columns:
        series = df["is_in_domain_target"]
        if series.dtype == bool:
            return series
        as_text = series.astype(str).str.lower().str.strip()
        return as_text.isin({"true", "1", "yes"})

    source_components = (
        df["source_dataset_components"].fillna("").astype(str).str.split(",").map(
            lambda xs: {x.strip() for x in xs if x.strip()}
        )
    )
    return pd.Series(
        [target in comps for target, comps in zip(df["target_dataset"].astype(str), source_components, strict=False)],
        index=df.index,
    )


def _ood_rows(df: pd.DataFrame) -> pd.DataFrame:
    in_domain = _is_in_domain_series(df)
    return df.loc[~in_domain].copy()


def _checkpoint_rollup(ood_df: pd.DataFrame) -> pd.DataFrame:
    ckpt = (
        ood_df.groupby(CKPT_KEYS, as_index=False)
        .agg(
            mean_ood_f1=("target_f1", "mean"),
            worst_target_f1=("target_f1", "min"),
            in_domain_f1=("in_domain_f1_mean", "mean"),
            latency_ms=("latency_mean_ms", "mean"),
            n_ood_targets=("target_dataset", "nunique"),
        )
    )
    ckpt["generalization_gap_f1"] = ckpt["in_domain_f1"] - ckpt["mean_ood_f1"]
    return ckpt


def _build_dataset_difficulty(ood_df: pd.DataFrame) -> pd.DataFrame:
    by_target = (
        ood_df.groupby("target_dataset", as_index=False)
        .agg(
            mean_target_f1=("target_f1", "mean"),
            std_target_f1=("target_f1", "std"),
            n_checkpoints=("cfg_id", "nunique"),
        )
    )
    by_target_model = (
        ood_df.groupby(["target_dataset", "source_model"], as_index=False)["target_f1"]
        .mean()
        .rename(columns={"target_f1": "model_mean_f1"})
    )
    extrema = (
        by_target_model.groupby("target_dataset", as_index=False)
        .agg(
            worst_model_f1=("model_mean_f1", "min"),
            best_model_f1=("model_mean_f1", "max"),
        )
    )
    out = by_target.merge(extrema, on="target_dataset", how="left")
    out["difficulty_rank"] = (
        out["mean_target_f1"].rank(method="min", ascending=True).astype(int)
    )
    out = out.sort_values(["difficulty_rank", "target_dataset"]).reset_index(drop=True)
    return out


def _build_best_models_by_mixed_source(ood_df: pd.DataFrame) -> pd.DataFrame:
    mixed = ood_df[ood_df["source_dataset_regime"].astype(str) == "mixed"].copy()
    out = (
        mixed.groupby(["source_dataset", "source_model"], as_index=False)
        .agg(
            mean_ood_f1=("target_f1", "mean"),
            worst_target_f1=("target_f1", "min"),
            target_coverage=("target_dataset", "nunique"),
            n_checkpoints=("cfg_id", "nunique"),
        )
    )
    if out.empty:
        out["rank_within_source_dataset"] = []
        return out
    out["rank_within_source_dataset"] = (
        out.groupby("source_dataset")["mean_ood_f1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out = out.sort_values(
        ["source_dataset", "rank_within_source_dataset", "source_model"]
    ).reset_index(drop=True)
    return out


def _build_regime_comparison(ckpt_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        ckpt_df.groupby(["source_dataset_regime", "source_model"], as_index=False)
        .agg(
            mean_ood_f1=("mean_ood_f1", "mean"),
            worst_target_f1=("worst_target_f1", "mean"),
            in_domain_f1=("in_domain_f1", "mean"),
            generalization_gap_f1=("generalization_gap_f1", "mean"),
            latency_ms=("latency_ms", "mean"),
            n_checkpoints=("cfg_id", "nunique"),
        )
    )
    solo_ref = (
        out[out["source_dataset_regime"] == "solo"][["source_model", "mean_ood_f1"]]
        .rename(columns={"mean_ood_f1": "solo_mean_ood_f1"})
    )
    out = out.merge(solo_ref, on="source_model", how="left")
    out["delta_vs_solo"] = out["mean_ood_f1"] - out["solo_mean_ood_f1"]
    out.loc[out["source_dataset_regime"] == "solo", "delta_vs_solo"] = 0.0
    out["rank_within_regime"] = (
        out.groupby("source_dataset_regime")["mean_ood_f1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out = out.sort_values(
        ["source_dataset_regime", "rank_within_regime", "source_model"]
    ).reset_index(drop=True)
    return out


def _build_legacy_summary(ckpt_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        ckpt_df.groupby(["source_dataset_regime", "source_model"], as_index=False)
        .agg(
            in_domain_macro_f1=("in_domain_f1", "mean"),
            cross_domain_macro_f1=("mean_ood_f1", "mean"),
            gap_f1=("generalization_gap_f1", "mean"),
            worst_target_f1=("worst_target_f1", "mean"),
            latency_ms=("latency_ms", "mean"),
            n_checkpoints=("cfg_id", "nunique"),
        )
        .sort_values(["source_dataset_regime", "cross_domain_macro_f1"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return out


def _to_tex_dataset_difficulty(df: pd.DataFrame) -> str:
    hardest_rank = int(df["difficulty_rank"].min()) if not df.empty else -1
    lines = [
        r"\begin{table*}[!tbph]",
        r"  \centering",
        r"  \caption{OOD target dataset difficulty ranking (lower mean F1 = harder).}",
        r"  \label{tab:dataset_difficulty_summary}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lcccccc}",
        r"    \hline",
        r"    \textbf{Target Dataset} & \textbf{Mean F1} & \textbf{Std F1} & \textbf{Worst Model F1} & \textbf{Best Model F1} & \textbf{N Ckpt} & \textbf{Rank} \\",
        r"    \hline",
    ]
    for _, r in df.iterrows():
        rank = int(r["difficulty_rank"])
        rank_cell = f"\\mathbf{{{rank}}}" if rank == hardest_rank else str(rank)
        lines.append(
            "    "
            + f"{_tex_escape(_pretty_dataset(r['target_dataset']))} & "
            + f"${_format_float(r['mean_target_f1'])}$ & "
            + f"${_format_float(r['std_target_f1'])}$ & "
            + f"$\\textit{{{_format_float(r['worst_model_f1'])}}}$ & "
            + f"${_format_float(r['best_model_f1'])}$ & "
            + f"{int(r['n_checkpoints'])} & "
            + f"${rank_cell}$ \\\\"
        )
    lines.extend([r"    \hline", r"  \end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _to_tex_best_mixed(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[!tbph]",
        r"  \centering",
        r"  \caption{Best models per mixed-source training setup on OOD targets (ranked by mean OOD F1).}",
        r"  \label{tab:best_models_by_mixed_source}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llccccc}",
        r"    \hline",
        r"    \textbf{Mixed Source} & \textbf{Model} & \textbf{Mean OOD F1} & \textbf{Worst-target F1} & \textbf{Target Coverage} & \textbf{N Ckpt} & \textbf{Rank} \\",
        r"    \hline",
    ]
    for source_dataset, block in df.groupby("source_dataset", sort=True):
        top_rank = int(block["rank_within_source_dataset"].min())
        for _, r in block.iterrows():
            rank = int(r["rank_within_source_dataset"])
            model_cell = (
                rf"\textbf{{{_tex_escape(r['source_model'])}}}"
                if rank == top_rank
                else _tex_escape(r["source_model"])
            )
            mean_cell = (
                rf"\mathbf{{{_format_float(r['mean_ood_f1'])}}}"
                if rank == top_rank
                else _format_float(r["mean_ood_f1"])
            )
            lines.append(
                "    "
                + f"{_tex_escape(_pretty_dataset(source_dataset))} & "
                + f"{model_cell} & "
                + f"${mean_cell}$ & "
                + f"$\\textit{{{_format_float(r['worst_target_f1'])}}}$ & "
                + f"{int(r['target_coverage'])} & "
                + f"{int(r['n_checkpoints'])} & "
                + f"{rank} \\\\"
            )
        lines.append(r"    \hline")
    lines.extend([r"  \end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _to_tex_regime_comparison(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[!tbph]",
        r"  \centering",
        r"  \caption{Solo vs mixed OOD transfer comparison by model.}",
        r"  \label{tab:regime_comparison_summary}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{llccccccc}",
        r"    \hline",
        r"    \textbf{Regime} & \textbf{Model} & \textbf{Mean OOD F1} & \textbf{Worst-target F1} & \textbf{In-domain F1} & \textbf{Gap F1} & \textbf{Delta vs Solo} & \textbf{N Ckpt} & \textbf{Rank} \\",
        r"    \hline",
    ]
    for regime, block in df.groupby("source_dataset_regime", sort=True):
        best_rank = int(block["rank_within_regime"].min())
        for _, r in block.iterrows():
            rank = int(r["rank_within_regime"])
            mean_cell = (
                rf"\mathbf{{{_format_float(r['mean_ood_f1'])}}}"
                if rank == best_rank
                else _format_float(r["mean_ood_f1"])
            )
            lines.append(
                "    "
                + f"{_tex_escape(regime)} & "
                + f"{_tex_escape(r['source_model'])} & "
                + f"${mean_cell}$ & "
                + f"$\\textit{{{_format_float(r['worst_target_f1'])}}}$ & "
                + f"${_format_float(r['in_domain_f1'])}$ & "
                + f"${_format_float(r['generalization_gap_f1'])}$ & "
                + f"${_format_float(r['delta_vs_solo'])}$ & "
                + f"{int(r['n_checkpoints'])} & "
                + f"{rank} \\\\"
            )
        lines.append(r"    \hline")
    lines.extend([r"  \end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _to_tex_legacy(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[!tbph]",
        r"  \centering",
        r"  \caption{Generalizability summary by model and training regime.}",
        r"  \label{tab:generalizability_summary}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{llccccc}",
        r"    \hline",
        r"    \textbf{Regime} & \textbf{Model} & \textbf{In-domain F1} & \textbf{Cross-domain F1} & \textbf{Gap F1} & \textbf{Worst-target F1} & \textbf{Latency (ms)} \\",
        r"    \hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            "    "
            + f"{_tex_escape(r['source_dataset_regime'])} & "
            + f"{_tex_escape(r['source_model'])} & "
            + f"${_format_float(r['in_domain_macro_f1'])}$ & "
            + f"${_format_float(r['cross_domain_macro_f1'])}$ & "
            + f"${_format_float(r['gap_f1'])}$ & "
            + f"${_format_float(r['worst_target_f1'])}$ & "
            + f"${_format_float(r['latency_ms'], 2)}$ \\\\"
        )
    lines.extend([r"    \hline", r"  \end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


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
        "source_dataset",
        "in_domain_f1_mean",
        "target_dataset",
        "target_f1",
        "latency_mean_ms",
        "is_in_domain_target",
        "source_dataset_components",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Master CSV missing required columns: {missing}")
    for c in ("in_domain_f1_mean", "target_f1", "latency_mean_ms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["in_domain_f1_mean", "target_f1"]).copy()

    out_dir = (OUT_DIR if OUT_DIR.is_absolute() else _REPO_ROOT / OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ood = _ood_rows(df)
    ckpt = _checkpoint_rollup(ood)
    dataset_difficulty = _build_dataset_difficulty(ood)
    best_mixed = _build_best_models_by_mixed_source(ood)
    regime_cmp = _build_regime_comparison(ckpt)
    legacy = _build_legacy_summary(ckpt)

    diff_csv = (DIFF_CSV if DIFF_CSV.is_absolute() else _REPO_ROOT / DIFF_CSV).resolve()
    mixed_csv = (MIXED_CSV if MIXED_CSV.is_absolute() else _REPO_ROOT / MIXED_CSV).resolve()
    regime_csv = (REGIME_CSV if REGIME_CSV.is_absolute() else _REPO_ROOT / REGIME_CSV).resolve()
    out_csv = (OUT_CSV if OUT_CSV.is_absolute() else _REPO_ROOT / OUT_CSV).resolve()
    diff_tex = (DIFF_TEX if DIFF_TEX.is_absolute() else _REPO_ROOT / DIFF_TEX).resolve()
    mixed_tex = (MIXED_TEX if MIXED_TEX.is_absolute() else _REPO_ROOT / MIXED_TEX).resolve()
    regime_tex = (REGIME_TEX if REGIME_TEX.is_absolute() else _REPO_ROOT / REGIME_TEX).resolve()
    out_tex = (OUT_TEX if OUT_TEX.is_absolute() else _REPO_ROOT / OUT_TEX).resolve()

    dataset_difficulty.to_csv(diff_csv, index=False)
    best_mixed.to_csv(mixed_csv, index=False)
    regime_cmp.to_csv(regime_csv, index=False)
    legacy.to_csv(out_csv, index=False)

    diff_tex.write_text(_to_tex_dataset_difficulty(dataset_difficulty), encoding="utf-8")
    mixed_tex.write_text(_to_tex_best_mixed(best_mixed), encoding="utf-8")
    regime_tex.write_text(_to_tex_regime_comparison(regime_cmp), encoding="utf-8")
    out_tex.write_text(_to_tex_legacy(legacy), encoding="utf-8")

    print(f"[OK] wrote {diff_csv}")
    print(f"[OK] wrote {mixed_csv}")
    print(f"[OK] wrote {regime_csv}")
    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {diff_tex}")
    print(f"[OK] wrote {mixed_tex}")
    print(f"[OK] wrote {regime_tex}")
    print(f"[OK] wrote {out_tex}")


if __name__ == "__main__":
    main()
