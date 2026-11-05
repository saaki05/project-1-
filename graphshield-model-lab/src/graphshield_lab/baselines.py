"""Feature-only baselines, optionally augmented with one-hop mean neighbour features.

The ``+agg`` variants answer a specific question: how much of a GNN's gain is explained by
simply handing a tabular model the neighbourhood-averaged features?
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

BASELINES = ("logreg", "rf", "hgb")


def neighbour_mean(x: np.ndarray, edge_index: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """Mean of neighbour features per node (zeros for isolated nodes)."""
    n = x.shape[0]
    src, dst = edge_index
    if symmetric:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    adj = sp.csr_matrix((np.ones(src.shape[0], dtype=np.float32), (dst, src)), shape=(n, n))
    adj.sum_duplicates()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return (adj @ x) / np.maximum(deg, 1.0)[:, None]


def make_baseline(name: str, seed: int, params: dict | None = None):
    """Build a sklearn model; ``params`` overrides the defaults (used by hyper-parameter search)."""
    p = params or {}
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed,
                                  C=p.get("C", 1.0))
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200), min_samples_leaf=p.get("min_samples_leaf", 1),
            max_features=p.get("max_features", "sqrt"), class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 200), learning_rate=p.get("learning_rate", 0.1),
            max_leaf_nodes=p.get("max_leaf_nodes", 31),
            l2_regularization=p.get("l2_regularization", 0.0), class_weight="balanced",
            random_state=seed)
    raise ValueError(f"unknown baseline {name!r}")
