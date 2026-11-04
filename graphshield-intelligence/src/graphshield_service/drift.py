"""Distribution-drift statistics implemented from first principles."""

from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index using reference-quantile bins.

    Common rule of thumb: < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 major shift.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (maximum ECDF gap)."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def drift_report(reference: np.ndarray, current: np.ndarray, top: int = 10) -> dict:
    """Per-feature PSI and KS; returns the ``top`` most drifted features."""
    if reference.shape[1] != current.shape[1]:
        raise ValueError("reference and current must have the same number of features")
    rows = [
        {"feature": j, "psi": psi(reference[:, j], current[:, j]),
         "ks": ks_statistic(reference[:, j], current[:, j])}
        for j in range(reference.shape[1])
    ]
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return {
        "mean_psi": float(np.mean([r["psi"] for r in rows])),
        "n_features_psi_gt_0.25": sum(r["psi"] > 0.25 for r in rows),
        "top": rows[:top],
    }
