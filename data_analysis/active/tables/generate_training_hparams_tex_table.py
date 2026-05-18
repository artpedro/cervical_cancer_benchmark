from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


# ============================================================
# CONFIG
# ============================================================
TRAIN_CONFIG_PY = Path("train_models.py")
OUTPUT_TEX = Path("workspace/analysis/training_hyperparameters_table.tex")


def _read_train_source(py_path: Path) -> str:
    resolved = py_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Training config not found: {resolved}")
    return resolved.read_text(encoding="utf-8")


def _same_or_join(values: list[Any]) -> str:
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    if not uniq:
        return "n/a"
    if len(uniq) == 1:
        return str(uniq[0])
    return ", ".join(str(v) for v in uniq)


def _literal_or_none(node: ast.AST) -> Any | None:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_global_constants(tree: ast.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            val = _literal_or_none(node.value)
            if val is not None:
                out[name] = val
    return out


def _extract_model_config_calls(tree: ast.Module) -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "ModelTrainConfig":
            continue
        rec: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            rec[kw.arg] = _literal_or_none(kw.value)
        cfgs.append(rec)
    return cfgs


def _escape_tex(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _generate_table(source: str) -> str:
    tree = ast.parse(source)
    consts = _extract_global_constants(tree)
    cfgs = _extract_model_config_calls(tree)
    if not cfgs:
        raise RuntimeError("Could not find ModelTrainConfig entries in training config.")

    epochs = _same_or_join([c.get("epochs", 25) for c in cfgs])
    lr = _same_or_join([c.get("lr", 5e-4) for c in cfgs])
    momentum = _same_or_join([c.get("momentum", 0.9) for c in cfgs])
    weight_decay = _same_or_join([c.get("weight_decay", 5e-3) for c in cfgs])
    milestones = _same_or_join([tuple(c.get("scheduler_milestones", [10, 20])) for c in cfgs])
    gamma = _same_or_join([c.get("scheduler_gamma", 0.1) for c in cfgs])
    pretrained = _same_or_join([c.get("pretrained", True) for c in cfgs])
    max_params_m = _same_or_join([c.get("max_params_m", 100.0) for c in cfgs])
    input_size = _same_or_join(
        [
            (c.get("load_kwargs") or {}).get("img_size", 224)
            if isinstance(c.get("load_kwargs"), dict)
            else 224
            for c in cfgs
        ]
    )
    datasets_list: list[str] = []
    for node in tree.body:
        value_node: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DATASETS"
        ):
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DATASETS":
            value_node = node.value

        if isinstance(value_node, ast.List):
            for elt in value_node.elts:
                if isinstance(elt, ast.Tuple) and elt.elts and isinstance(elt.elts[0], ast.Constant):
                    ds_name = elt.elts[0].value
                    if isinstance(ds_name, str):
                        datasets_list.append(ds_name)
            break
    datasets = ", ".join(datasets_list)

    rows: list[tuple[str, str]] = [
        ("Random seed", str(consts.get("SEED", "n/a"))),
        ("Datasets", datasets or "n/a"),
        ("Cross-validation folds", str(consts.get("NUM_FOLDS", "n/a"))),
        ("Class-imbalance handling", str(consts.get("BALANCE_MODE", "n/a"))),
        ("Epochs", epochs),
        ("Batch size", str(consts.get("BATCH_SIZE", "n/a"))),
        ("Input image size", str(input_size)),
        ("Optimizer", "SGD"),
        ("Learning rate", lr),
        ("Momentum", momentum),
        ("Weight decay", weight_decay),
        ("LR scheduler", "MultiStepLR"),
        ("Scheduler milestones", milestones),
        ("Scheduler gamma", gamma),
        ("Loss function", "CrossEntropyLoss (class-weighted)"),
        ("Mixed precision (AMP)", str(consts.get("USE_AMP", "n/a"))),
        ("Pretrained backbone", pretrained),
        ("Max params (M)", max_params_m),
        ("Num workers", str(consts.get("NUM_WORKERS", "n/a"))),
    ]

    lines: list[str] = []
    lines.append(r"\begin{table}[!tbph]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Shared training hyperparameters used across all evaluated models.}"
    )
    lines.append(r"  \label{tab:training_hparams}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \hline")
    lines.append(r"    \textbf{Hyperparameter} & \textbf{Value} \\")
    lines.append(r"    \hline")
    for k, v in rows:
        lines.append(f"    {_escape_tex(k)} & {_escape_tex(v)} \\\\")
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    source = _read_train_source(TRAIN_CONFIG_PY)
    tex = _generate_table(source)
    out = OUTPUT_TEX.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

