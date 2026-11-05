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
    st = sub.add_parser("static", help="compare models on a graph without time information")
    st.add_argument("--amazon", default=None, help="path to Amazon.mat")
    st.add_argument("--relation", default="upu", choices=["homo", "upu", "usu", "uvu"])
    st.add_argument("--data", default=None, help="or a graphshield-data directory")
    st.add_argument("--models", nargs="+",
                    default=["logreg", "rf", "hgb", "hgb+agg", "mlp", "gcn", "sage", "gat"])
    st.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    st.add_argument("--epochs", type=int, default=100)
    st.add_argument("--tune-trials", type=int, default=0,
                    help="random-search trials per model on a reserved tuning pool (0 = off)")
    st.add_argument("--pool-fraction", type=float, default=0.2)
    st.add_argument("--out", default=None)
    dr = sub.add_parser("drift", help="static vs rolling-window retraining, block by block")
    dr.add_argument("--data", required=True)
    dr.add_argument("--models", nargs="+", default=["hgb", "gcn", "sage"])
    dr.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    dr.add_argument("--window", type=int, default=10)
    dr.add_argument("--horizon", type=int, default=3)
    dr.add_argument("--static-train-end", type=int, default=None)
    dr.add_argument("--static-val-end", type=int, default=None)
    dr.add_argument("--epochs", type=int, default=200)
    dr.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.command == "models":
        print("\n".join(available_models()))
        return 0
    if args.command == "static":
        from graphshield_data.amazon import load_amazon

        from graphshield_lab.static import render_static, run_static_study

        if bool(args.amazon) == bool(args.data):
            raise SystemExit("give exactly one of --amazon or --data")
        graph = (load_amazon(args.amazon, args.relation) if args.amazon
                 else load_graph(args.data))
        base = TrainConfig(epochs=args.epochs)
        extra = {}
        if args.tune_trials:
            from graphshield_lab.tune import apply_best, split_tuning_pool, tune_model

            tune_graph, graph = split_tuning_pool(graph, args.pool_fraction)
            cache = Path(args.out + ".tuning.json") if args.out else None
            done = json.loads(cache.read_text()) if cache and cache.exists() else {}
            tuned, cfgs, params = {}, {}, {}
            for name in args.models:
                key = name.removesuffix("+agg")
                if key not in done:  # +agg variants reuse the base model's search
                    done[key] = tune_model(tune_graph, key if key in ("mlp", "gcn", "sage", "gat")
                                           else name, args.tune_trials, base_cfg=base)
                    if cache:  # keep progress: a crash must not lose hours of search
                        cache.write_text(json.dumps(done))
                tuned[name] = done[key]
                cfgs[name], params[name] = apply_best(name, done[key], base)
                cfgs[name] = cfgs[name] if params[name] is None else base
            extra = {"model_cfgs": cfgs, "model_params": {k: v for k, v in params.items() if v}}
        res = run_static_study(graph, args.models, args.seeds, base, **extra)
        if args.tune_trials:
            res["tuning"] = {k: {"best": v["best"], "best_val_pr_auc": v["best_score"],
                                 "default_val_pr_auc": v["default_score"],
                                 "n_trials": v["n_trials"]} for k, v in tuned.items()}
            res["tuning_pool_fraction"] = args.pool_fraction
        print(render_static(res))
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return 0

    graph = load_graph(args.data)
    if args.command == "drift":
        from graphshield_lab.drift import drift_study

        res = drift_study(graph, args.models, args.seeds, args.window, args.horizon,
                          args.static_train_end, args.static_val_end,
                          TrainConfig(epochs=args.epochs))
        for model, s in res["summary"].items():
            print(f"{model:<10} static {s['mean_static']:.3f}  rolling {s['mean_rolling']:.3f}  "
                  f"gain {s['mean_gain']:+.3f}  rolling wins {s['blocks_where_rolling_wins']}/"
                  f"{s['blocks']} blocks")
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return 0
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
