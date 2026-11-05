"""Model comparison on graphs without time information.

Each seed draws a new stratified train/validation/test split of the labelled nodes and
trains every model on that same split, so models are compared *paired* within a seed.
Training is transductive: the whole graph structure and all node features are visible, but
only training labels are used. Early stopping and the F1 threshold use the validation part.
"""

from __future__ import annotations

import numpy as np
from graphshield_data import GraphData, audit_split, stratified_split

from graphshield_lab.runner import TrainConfig, evaluate, run_model

METRICS = ("pr_auc", "roc_auc", "f1", "mcc", "precision_at_100", "ece")


def _bootstrap_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def run_static_study(graph: GraphData, models: list[str], seeds: list[int],
                     cfg: TrainConfig | None = None, fractions=(0.4, 0.2, 0.4),
                     references: tuple[str, ...] = ("hgb", "hgb+agg"),
                     model_cfgs: dict | None = None, model_params: dict | None = None) -> dict:
    """Run every model on a fresh stratified split per seed; summarise with paired contrasts.

    ``model_cfgs`` / ``model_params`` optionally give a tuned ``TrainConfig`` (neural models) or
    sklearn parameter dict (tabular models) per model name.
    """
    cfg = cfg or TrainConfig()
    model_cfgs, model_params = model_cfgs or {}, model_params or {}
    per_seed: dict[str, list[dict]] = {m: [] for m in models}
    for seed in seeds:
        split = stratified_split(graph, fractions, seed)
        report = audit_split(graph, split)
        if not report.ok:
            raise RuntimeError("split failed the audit:\n" + report.format())
        for name in models:
            pred = run_model(name, graph, split, seed, model_cfgs.get(name, cfg),
                             model_params.get(name))
            per_seed[name].append(evaluate(pred, graph, split))

    summary = {}
    for name, rows in per_seed.items():
        summary[name] = {m: {"mean": float(np.mean([r[m] for r in rows])),
                             "std": float(np.std([r[m] for r in rows]))} for m in METRICS}
    contrasts = {}
    for ref in references:
        if ref not in per_seed:
            continue
        contrasts[ref] = {}
        base = np.array([r["pr_auc"] for r in per_seed[ref]])
        for name, rows in per_seed.items():
            if name == ref:
                continue
            diff = np.array([r["pr_auc"] for r in rows]) - base
            lo, hi = _bootstrap_ci(diff)
            contrasts[ref][name] = {"mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                                    "wins": int((diff > 0).sum()), "n": len(diff)}
    labelled = graph.labels >= 0
    return {"n_nodes": graph.num_nodes, "n_edges": graph.num_edges,
            "n_labelled": int(labelled.sum()),
            "fraud_rate": float((graph.labels[labelled] == 1).mean()),
            "fractions": list(fractions), "seeds": list(seeds), "models": models,
            "config": {"hidden": cfg.hidden, "epochs": cfg.epochs, "patience": cfg.patience,
                       "lr": cfg.lr, "dropout": cfg.dropout},
            "tuned": bool(model_cfgs or model_params),
            "summary": summary, "pr_auc_contrasts": contrasts, "per_seed": per_seed}


def render_static(res: dict, title: str = "GraphShield static-graph study") -> str:
    lines = [f"# {title} (generated from the result JSON)", "",
             f"{res['n_nodes']} nodes, {res['n_edges']} directed edges, {res['n_labelled']} "
             f"labelled nodes ({res['fraud_rate']:.1%} fraud). Split fractions "
             f"{res['fractions']} (train/val/test), a fresh stratified split per seed; seeds "
             f"{res['seeds']}. Mean ± std over seeds.", "",
             "| model | PR-AUC | ROC-AUC | F1 | MCC | P@100 | ECE |",
             "|---|---|---|---|---|---|---|"]
    for name, s in res["summary"].items():
        cells = " | ".join(f"{s[m]['mean']:.3f} ± {s[m]['std']:.3f}" for m in METRICS)
        lines.append(f"| {name} | {cells} |")
    for ref, rows in res["pr_auc_contrasts"].items():
        lines += ["", f"## PR-AUC difference to `{ref}` (paired by seed, bootstrap 95% CI of the "
                  "mean over seeds)", "", "| model | mean diff | 95% CI | seeds better |",
                  "|---|---|---|---|"]
        for name, c in rows.items():
            lines.append(f"| {name} | {c['mean_diff']:+.3f} | [{c['ci_low']:+.3f}, "
                         f"{c['ci_high']:+.3f}] | {c['wins']}/{c['n']} |")
    return "\n".join(lines) + "\n"
