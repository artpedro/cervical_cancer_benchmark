from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


# ============================================================
# CONFIG
# ============================================================
PROFILE_SCRIPT = Path("data_analysis/active/profile_checkpoint_efficiency.py")
OUTPUT_TEX = Path("workspace/analysis/efficiency_profile/inference_efficiency_settings_table.tex")


def _read_source(path: Path) -> str:
    p = path.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Profile script not found: {p}")
    return p.read_text(encoding="utf-8")


def _literal_or_none(node: ast.AST) -> Any | None:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_constants(tree: ast.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            val = _literal_or_none(node.value)
            if val is not None:
                out[name] = val
    return out


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _generate_table(source: str) -> str:
    tree = ast.parse(source)
    c = _extract_constants(tree)

    latency_basis = str(c.get("LATENCY_MS_REPORT", "n/a"))
    basis_pretty = "ms/image" if latency_basis == "image" else "ms/batch" if latency_basis == "batch" else latency_basis

    rows: list[tuple[str, str]] = [
        ("Seed", str(c.get("SEED", "n/a"))),
        ("Fold count used by dataset scanner", str(c.get("NUM_FOLDS", "n/a"))),
        ("Test split ratio (scanner)", str(c.get("TEST_SIZE", "n/a"))),
        ("Device setting", str(c.get("DEVICE", "auto"))),
        ("Latency input tensor shape", f"(B, 3, H, W) with B={c.get('LATENCY_BATCH_SIZE', 'n/a')}, H=W"),
        ("Latency batch size (B)", str(c.get("LATENCY_BATCH_SIZE", "n/a"))),
        ("Default input spatial size", str(c.get("LATENCY_INPUT_SPATIAL", "n/a"))),
        ("Warmup iterations", str(c.get("LATENCY_WARMUP_ITERATIONS", "n/a"))),
        ("Timed repetitions", str(c.get("LATENCY_TIMED_REPETITIONS", "n/a"))),
        ("Reported latency basis", basis_pretty),
        ("Reported latency statistic", "median of timed passes"),
        ("Model mode during timing", "model.eval() + torch.inference_mode()"),
        ("CUDA timing protocol", "synchronize before/after each timed pass; CUDA events"),
        ("CPU timing protocol", "time.perf\\_counter around forward pass"),
        ("Memory tracked", "mean and max peak allocated memory across timed passes"),
        ("Complexity metrics", "parameter count, MACs, FLOPs (FLOPs = 2 x MACs)"),
    ]

    lines: list[str] = []
    lines.append(r"\begin{table}[!tbph]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Inference-efficiency profiling setup used to compute latency, memory, and complexity metrics.}"
    )
    lines.append(r"  \label{tab:inference_efficiency_setup}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \hline")
    lines.append(r"    \textbf{Parameter} & \textbf{Value} \\")
    lines.append(r"    \hline")
    for k, v in rows:
        lines.append(f"    {_tex_escape(k)} & {_tex_escape(v)} \\\\")
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    src = _read_source(PROFILE_SCRIPT)
    tex = _generate_table(src)
    out = OUTPUT_TEX.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

