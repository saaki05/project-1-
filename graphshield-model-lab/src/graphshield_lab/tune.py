"""Random hyper-parameter search with a reserved tuning pool (no leakage into evaluation).

A fraction of the labelled nodes is set aside as a *tuning pool*. Search trials train and
validate only on labels inside that pool; the evaluation graph has the pool's labels hidden,
so no node used to choose hyper-parameters is ever a test node. Every model family, tabular
and graph, gets the same number of random trials.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from graphshield_data import GraphData, stratified_split

from graphshield_lab import metrics as M
from graphshield_lab.runner import TrainConfig, run_model

TABULAR_SPACES = {
    "logreg": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    "rf": {"n_estimators": [100, 300], "min_samples_leaf": [1, 2, 5, 10],
           "max_features": ["sqrt", 0.3, 0.5]},
    "hgb": {"learning_rate": [0.03, 0.05, 0.1, 0.2], "max_iter": [100, 200, 400],
            "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.1, 1.0]},
}
NN_SPACE = {"hidden": [32, 64, 128], "num_layers": [1, 2, 3], "lr": [1e-3, 3e-3, 1e-2],
            "dropout": [0.0, 0.3, 0.5], "weight_decay": [0.0, 5e-4, 5e-3]}
NN_MODELS = ("mlp", "gcn", "sage", "gat")


def split_tuning_pool(graph: GraphData, fraction: float = 0.2, seed: int = 123):
    """Return ``(tuning_graph, evaluation_graph)``.

    ``tuning_graph`` keeps labels only inside the pool; ``evaluation_graph`` hides the pool's
    labels (sets them to unknown) so evaluation splits never include pool nodes.
    """
    labelled = np.flatnonzero(graph.labels >= 0)
    rng = np.random.default_rng(seed)
    pool = np.zeros(graph.num_nodes, dtype=bool)
    for cls in (0, 1):
        idx = rng.permutation(labelled[graph.labels[labelled] == cls])
        pool[idx[: max(3, int(round(fraction * len(idx))))]] = True
    tune_labels = np.where(pool, graph.labels, -1).astype(graph.labels.dtype)
    eval_labels = np.where(pool, -1, graph.labels).astype(graph.labels.dtype)
    return (replace(graph, labels=tune_labels), replace(graph, labels=eval_labels))


def _sample(space: dict, rng: np.random.Generator) -> dict:
    return {k: v[int(rng.integers(len(v)))] for k, v in space.items()}


def tune_model(graph: GraphData, name: str, n_trials: int = 12, tuning_seeds=(100, 101),
               base_cfg: TrainConfig | None = None, seed: int = 0) -> dict:
    """Random search for ``name`` on a tuning graph; the score is the mean validation PR-AUC.

    ``name`` may carry the ``+agg`` suffix; its search space is that of the base model.
    Returns ``{"best": {...}, "best_score": float, "default_score": float, "trials": [...]}``.
    """
    base = name.removesuffix("+agg")
    nn_model = base in NN_MODELS
    space = NN_SPACE if nn_model else TABULAR_SPACES[base]
    cfg0 = base_cfg or TrainConfig(epochs=100, patience=20)
    rng = np.random.default_rng(seed)
    splits = [stratified_split(graph, (0.5, 0.25, 0.25), s) for s in tuning_seeds]

    def score(candidate: dict) -> float:
        vals = []
        for sp, s in zip(splits, tuning_seeds, strict=True):
            if nn_model:
                if base == "gat" and candidate.get("hidden", cfg0.hidden) % cfg0.heads:
                    return -1.0
                pred = run_model(base, graph, sp, s, replace(cfg0, **candidate))
            else:
                pred = run_model(name, graph, sp, s, cfg0, candidate)
            y = graph.labels[sp.val].astype(int)
            vals.append(M.pr_auc(y, pred.val_prob))
        return float(np.mean(vals))

    default = {} if not nn_model else {}
    trials = []
    for _ in range(n_trials):
        cand = _sample(space, rng)
        trials.append({"params": cand, "val_pr_auc": score(cand)})
    best = max(trials, key=lambda t: t["val_pr_auc"])
    return {"model": name, "best": best["params"], "best_score": best["val_pr_auc"],
            "default_score": score(default), "n_trials": n_trials, "trials": trials}


def apply_best(name: str, tuned: dict, base_cfg: TrainConfig) -> tuple[TrainConfig, dict | None]:
    """Turn a tuning result into ``(TrainConfig, tabular_params)`` for ``run_model``."""
    if name.removesuffix("+agg") in NN_MODELS:
        return replace(base_cfg, **tuned["best"]), None
    return base_cfg, tuned["best"]
