"""Shared helpers for RQ statistical analysis scripts.

Kept dependency-light: only numpy / pandas / scipy. Provides:
- Benjamini-Hochberg FDR correction
- Bootstrap CI for the mean and arbitrary statistics
- Cohen's d for paired and independent samples
- Markdown rendering of small tables (no external `tabulate`)
- Helpers to apply user-facing filters (model whitelist/exclude lists, regime
  inclusion) consistently across scripts
- Architecture family classifier shared by RQ3 and RQ4 stratifications
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TRANSFORMER_TOKENS: tuple[str, ...] = (
    "vit",
    "former",
    "iformer",
    "levit",
    "fastvit",
    "mobilevit",
    "eat",
)


def model_family(model_name: str) -> str:
    """Classify a model into ``transformer`` or ``cnn`` by tokens in the name."""
    name = str(model_name).lower()
    if any(token in name for token in TRANSFORMER_TOKENS):
        return "transformer"
    return "cnn"


def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; preserves NaNs."""
    p = np.asarray(list(pvals), dtype=float)
    out = np.full_like(p, np.nan)
    valid_mask = ~np.isnan(p)
    valid = p[valid_mask]
    if valid.size == 0:
        return out
    order = np.argsort(valid)
    ranked = valid[order]
    n = ranked.size
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[valid_mask] = restored
    return out


def bootstrap_ci_mean(values: np.ndarray, *, n_boot: int = 10_000,
                      ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def bootstrap_ci_statistic(values: np.ndarray, statistic, *, n_boot: int = 10_000,
                           ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = float(statistic(sample))
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def bootstrap_ci_paired_statistic(x: np.ndarray, y: np.ndarray, statistic,
                                  *, n_boot: int = 10_000, ci: float = 0.95,
                                  seed: int = 42) -> tuple[float, float]:
    """Resample paired (x, y) jointly to compute a CI for ``statistic(x, y)``."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size or x_arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = x_arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(statistic(x_arr[idx], y_arr[idx]))
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]
    if a_arr.size < 2 or b_arr.size < 2:
        return float("nan")
    var_a = float(np.var(a_arr, ddof=1))
    var_b = float(np.var(b_arr, ddof=1))
    n_a = a_arr.size
    n_b = b_arr.size
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled)


def df_to_markdown(df: pd.DataFrame, *, float_fmt: str = "{:.4f}") -> str:
    if df.empty:
        return "(no rows)"

    def _fmt(v: object) -> str:
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            if np.isnan(v):
                return "n/a"
            return float_fmt.format(v)
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (np.floating,)):
            if np.isnan(v):
                return "n/a"
            return float_fmt.format(float(v))
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[_fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    head = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|"
    body = ["|" + "|".join(f" {v:<{w}} " for v, w in zip(row, widths)) + "|" for row in rows]
    return "\n".join([head, sep, *body])


def apply_filters(
    df: pd.DataFrame,
    *,
    models_include: Sequence[str] | None = None,
    models_exclude: Sequence[str] = (),
    regimes_include: Sequence[str] | None = None,
    source_datasets_include: Sequence[str] | None = None,
    source_datasets_exclude: Sequence[str] = (),
    target_datasets_include: Sequence[str] | None = None,
    target_datasets_exclude: Sequence[str] = (),
    model_col: str = "source_model",
    regime_col: str = "source_dataset_regime",
    source_dataset_col: str = "source_dataset",
    target_dataset_col: str = "target_dataset",
) -> pd.DataFrame:
    """Apply consistent inclusion/exclusion filters used by all RQ scripts.

    Empty-or-None include lists mean "no restriction". All matches are by
    exact string equality on the corresponding column.
    """
    out = df.copy()
    if models_include is not None and len(models_include) > 0:
        out = out[out[model_col].astype(str).isin(set(models_include))]
    if models_exclude:
        out = out[~out[model_col].astype(str).isin(set(models_exclude))]
    if regimes_include is not None and len(regimes_include) > 0:
        out = out[out[regime_col].astype(str).isin(set(regimes_include))]
    if source_datasets_include is not None and len(source_datasets_include) > 0:
        out = out[out[source_dataset_col].astype(str).isin(set(source_datasets_include))]
    if source_datasets_exclude:
        out = out[~out[source_dataset_col].astype(str).isin(set(source_datasets_exclude))]
    if target_datasets_include is not None and len(target_datasets_include) > 0 and target_dataset_col in out.columns:
        out = out[out[target_dataset_col].astype(str).isin(set(target_datasets_include))]
    if target_datasets_exclude and target_dataset_col in out.columns:
        out = out[~out[target_dataset_col].astype(str).isin(set(target_datasets_exclude))]
    return out.reset_index(drop=True)


def describe_filters(*, models_include: Sequence[str] | None, models_exclude: Sequence[str],
                     regimes_include: Sequence[str] | None,
                     source_datasets_include: Sequence[str] | None,
                     source_datasets_exclude: Sequence[str],
                     target_datasets_include: Sequence[str] | None,
                     target_datasets_exclude: Sequence[str]) -> list[str]:
    """Render filter selections as markdown bullet lines for decision summaries."""
    def _fmt(name: str, values: Sequence[str] | None, *, default_all: bool) -> str:
        if values is None or len(values) == 0:
            return f"- {name}: {'all' if default_all else 'none'}"
        return f"- {name}: {', '.join(values)}"

    return [
        _fmt("Models included", models_include, default_all=True),
        _fmt("Models excluded", models_exclude, default_all=False),
        _fmt("Regimes included", regimes_include, default_all=True),
        _fmt("Source datasets included", source_datasets_include, default_all=True),
        _fmt("Source datasets excluded", source_datasets_exclude, default_all=False),
        _fmt("Target datasets included", target_datasets_include, default_all=True),
        _fmt("Target datasets excluded", target_datasets_exclude, default_all=False),
    ]


def cluster_robust_ols(y: np.ndarray, X: np.ndarray,
                       clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form OLS with cluster-robust covariance (CR1 sandwich).

    Returns (beta, cov_beta) where ``cov_beta`` uses the small-sample CR1
    correction ``G / (G - 1) * (n - 1) / (n - k)`` over ``G`` clusters,
    ``n`` rows, ``k`` parameters.
    """
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    n, k = X_arr.shape
    XtX_inv = np.linalg.pinv(X_arr.T @ X_arr)
    beta = XtX_inv @ X_arr.T @ y_arr
    resid = y_arr - X_arr @ beta
    unique = np.unique(clusters)
    G = unique.size
    meat = np.zeros((k, k), dtype=float)
    for cl in unique:
        mask = clusters == cl
        u = (X_arr[mask].T @ resid[mask]).reshape(-1, 1)
        meat += u @ u.T
    cov = XtX_inv @ meat @ XtX_inv
    cov *= (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return beta, cov
