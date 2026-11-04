"""Metrics for rare-positive detection, with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def f1_at(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = p >= threshold
    tp = float((pred & (y == 1)).sum())
    denom = 2 * tp + float((pred & (y == 0)).sum()) + float((~pred & (y == 1)).sum())
    return 0.0 if denom == 0 else 2 * tp / denom


def mcc_at(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p >= threshold).astype(int)
    if pred.min() == pred.max():
        return 0.0
    return float(matthews_corrcoef(y, pred))


def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Fraction of true positives among the ``k`` highest-scored nodes (analyst alert budget)."""
    k = min(k, len(y))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean())


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on the given (validation) data."""
    candidates = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 201)))
    scores = [f1_at(y, p, t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def bootstrap_ci(
    metric: MetricFn,
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval; resamples lacking both classes are skipped."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], p[idx]))
    if not values:
        raise ValueError("no valid bootstrap resamples")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(
    metric: MetricFn,
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of metric(A) - metric(B) over the same evaluation nodes.

    Returns the observed difference, a 95% interval and a two-sided bootstrap p-value
    (twice the smaller tail mass at zero).
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    d = np.asarray(diffs)
    lo, hi = np.quantile(d, [0.025, 0.975])
    p_value = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {
        "diff": float(metric(y, p_a) - metric(y, p_b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value),
    }


def per_timestep(
    metric: MetricFn, y: np.ndarray, p: np.ndarray, timestep: np.ndarray
) -> dict[int, float]:
    """Metric at each time step; steps with a single class are omitted."""
    out: dict[int, float] = {}
    for t in np.unique(timestep):
        m = timestep == t
        if y[m].min() != y[m].max():
            out[int(t)] = metric(y[m], p[m])
    return out
