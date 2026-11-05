"""Static versus rolling-window retraining under temporal drift.

The *static* protocol trains once on early time steps and is evaluated on every later block.
The *rolling* protocol retrains before each block on the most recent ``window`` steps. Both
are evaluated on exactly the same test nodes (each block), so the per-block PR-AUC
difference isolates the value of retraining. Every training split passes the leakage audit.
"""

from __future__ import annotations

import numpy as np
from graphshield_data import GraphData, audit_split, rolling_windows, temporal_split

from graphshield_lab import metrics as M
from graphshield_lab.runner import TrainConfig, run_model


def _block_pr_auc(graph: GraphData, test_mask: np.ndarray, probs: np.ndarray, lo: int, hi: int):
    idx = np.flatnonzero(test_mask)
    t = graph.timestep[idx]
    sel = (t >= lo) & (t <= hi)
    y = graph.labels[idx][sel].astype(int)
    if sel.sum() == 0 or y.min() == y.max():
        return None, int(y.sum())
    return M.pr_auc(y, probs[sel]), int(y.sum())


def drift_study(graph: GraphData, models: list[str], seeds: list[int], window: int = 10,
                horizon: int = 3, static_train_end: int | None = None,
                static_val_end: int | None = None, cfg: TrainConfig | None = None) -> dict:
    """Compare static and rolling retraining block by block; returns rows and a summary."""
    cfg = cfg or TrainConfig()
    t_min, t_max = int(graph.timestep.min()), int(graph.timestep.max())
    train_end = static_train_end if static_train_end is not None else t_min + window - 3
    val_end = static_val_end if static_val_end is not None else train_end + 2
    static = temporal_split(graph, (t_min, train_end), (train_end + 1, val_end),
                            (val_end + 1, t_max), name="static")
    if not audit_split(graph, static).ok:
        raise RuntimeError("static split failed the leakage audit")

    blocks = [w for w in rolling_windows(graph, window, horizon, val_steps=2, stride=horizon)
              if int(graph.timestep[w.test].min()) > val_end]
    rows = []
    for model in models:
        static_preds = [run_model(model, graph, static, s, cfg) for s in seeds]
        static_prob = np.mean([p.test_prob for p in static_preds], axis=0)
        for w in blocks:
            lo, hi = int(graph.timestep[w.test].min()), int(graph.timestep[w.test].max())
            if not audit_split(graph, w).ok:
                rows.append({"model": model, "block": [lo, hi], "skipped": "audit failed"})
                continue
            roll_prob = np.mean([run_model(model, graph, w, s, cfg).test_prob for s in seeds],
                                axis=0)
            roll, n_pos = _block_pr_auc(graph, w.test, roll_prob, lo, hi)
            # static predictions cover the whole static test range; select the same block
            stat, _ = _block_pr_auc(graph, static.test, static_prob, lo, hi)
            if roll is None or stat is None:
                rows.append({"model": model, "block": [lo, hi], "skipped": "single class"})
                continue
            rows.append({"model": model, "block": [lo, hi], "n_illicit": n_pos,
                         "static_pr_auc": stat, "rolling_pr_auc": roll, "gain": roll - stat})
    summary = {}
    for model in models:
        ok = [r for r in rows if r["model"] == model and "gain" in r]
        if ok:
            gains = np.array([r["gain"] for r in ok])
            summary[model] = {"blocks": len(ok), "mean_static": float(np.mean(
                [r["static_pr_auc"] for r in ok])),
                "mean_rolling": float(np.mean([r["rolling_pr_auc"] for r in ok])),
                "mean_gain": float(gains.mean()),
                "blocks_where_rolling_wins": int((gains > 0).sum())}
    return {"window": window, "horizon": horizon, "static_train_end": train_end,
            "static_val_end": val_end, "seeds": list(seeds), "rows": rows, "summary": summary}
