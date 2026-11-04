"""``graphshield-lab run``: evaluate several models over several seeds on one temporal split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from graphshield_data import audit_split, temporal_split
from graphshield_data.store import load_graph

from graphshield_lab import metrics as M
from graphshield_lab.runner import TrainConfig, available_models, config_dict, evaluate, run_model


def run_experiment(graph, split, models, seeds, cfg=None) -> dict:
    """Run every model for every seed; return per-seed metrics and a summary table."""
    cfg = cfg or TrainConfig()
    report = audit_split(graph, split)
    if not report.ok:
        raise RuntimeError("split failed the leakage audit:\n" + report.format())

    y_test = graph.labels[split.test].astype(int)
    per_seed: dict[str, list[dict]] = {}
    mean_probs: dict[str, np.ndarray] = {}
    for name in models:
        runs = [run_model(name, graph, split, s, cfg) for s in seeds]
        per_seed[name] = [evaluate(r, graph, split) for r in runs]
        mean_probs[name] = np.mean([r.test_prob for r in runs], axis=0)

    summary = {}
    for name, rows in per_seed.items():
        summary[name] = {
            k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]
        }
        lo, hi = M.bootstrap_ci(M.pr_auc, y_test, mean_probs[name], n_boot=500)
        summary[name]["pr_auc_seed_averaged_ci95"] = [lo, hi]
    return {
        "config": config_dict(cfg),
        "seeds": list(seeds),
        "n_test": int(len(y_test)),
        "test_illicit_rate": float(y_test.mean()),
        "per_seed": per_seed,
        "summary": summary,
        "_mean_probs": {k: v.tolist() for k, v in mean_probs.items()},
    }


def _format_table(result: dict) -> str:
    lines = [f"{'model':<14}{'PR-AUC':>16}{'F1':>16}{'P@100':>10}{'ECE':>8}"]
    for name, s in result["summary"].items():
        lines.append(
            f"{name:<14}{s['pr_auc']['mean']:>9.3f}±{s['pr_auc']['std']:.3f}"
            f"{s['f1']['mean']:>9.3f}±{s['f1']['std']:.3f}"
            f"{s['precision_at_100']['mean']:>10.3f}{s['ece']['mean']:>8.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="graphshield-lab")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--data", required=True, help="directory written by graphshield-data")
    run.add_argument("--train-end", type=int, default=29)
    run.add_argument("--val-end", type=int, default=34)
    run.add_argument("--models", nargs="+", default=["logreg", "rf", "gcn", "sage", "gat"])
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--epochs", type=int, default=200)
    run.add_argument("--out", default=None)
    sub.add_parser("models")
    args = ap.parse_args(argv)

    if args.command == "models":
        print("\n".join(available_models()))
        return 0

    graph = load_graph(args.data)
    split = temporal_split(
        graph,
        train=(int(graph.timestep.min()), args.train_end),
        val=(args.train_end + 1, args.val_end),
        test=(args.val_end + 1, int(graph.timestep.max())),
    )
    result = run_experiment(graph, split, args.models, args.seeds, TrainConfig(epochs=args.epochs))
    print(_format_table(result))
    if args.out:
        result.pop("_mean_probs")
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
