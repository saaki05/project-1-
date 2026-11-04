"""Command-line entry point: ``graphshield-data``."""

from __future__ import annotations

import argparse
import sys

from graphshield_data.audit import audit_split
from graphshield_data.elliptic import load_elliptic, write_elliptic
from graphshield_data.splits import temporal_split
from graphshield_data.store import load_graph, save_graph
from graphshield_data.synthetic import make_synthetic_graph


def _cmd_convert(args: argparse.Namespace) -> int:
    graph = load_elliptic(args.raw)
    manifest = save_graph(graph, args.out)
    print(f"wrote {manifest['num_nodes']} nodes / {manifest['num_edges']} edges to {args.out}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    graph = make_synthetic_graph(
        num_steps=args.steps,
        nodes_per_step=args.nodes_per_step,
        shift_step=args.shift_step,
        seed=args.seed,
    )
    save_graph(graph, args.out)
    if args.raw_out:
        write_elliptic(graph, args.raw_out)
    print(f"wrote synthetic graph ({graph.num_nodes} nodes) to {args.out}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    graph = load_graph(args.data)
    last = int(graph.timestep.max())
    train_end, val_end = args.train_end, args.val_end
    split = temporal_split(
        graph,
        train=(int(graph.timestep.min()), train_end),
        val=(train_end + 1, val_end),
        test=(val_end + 1, last),
    )
    report = audit_split(graph, split)
    print(report.format())
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphshield-data")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert raw Elliptic CSVs to versioned Parquet")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("synth", help="generate a synthetic graph (for tests only)")
    p.add_argument("--out", required=True)
    p.add_argument("--raw-out", default=None, help="also write raw Elliptic-style CSVs")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--nodes-per-step", type=int, default=200)
    p.add_argument("--shift-step", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_synth)

    p = sub.add_parser("audit", help="audit a temporal split of a stored graph")
    p.add_argument("--data", required=True)
    p.add_argument("--train-end", type=int, default=29)
    p.add_argument("--val-end", type=int, default=34)
    p.set_defaults(func=_cmd_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
