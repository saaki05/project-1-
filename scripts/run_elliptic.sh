#!/usr/bin/env bash
# Reproduce the GraphShield study on the real Elliptic data.
# Usage: scripts/run_elliptic.sh /path/to/elliptic_bitcoin_dataset results
# Download the dataset yourself from its official source and read its license first.
set -euo pipefail
RAW="${1:?directory with elliptic_txs_features.csv, _classes.csv, _edgelist.csv}"
OUT="${2:-results}"
mkdir -p "$OUT" data

graphshield-data convert --raw "$RAW" --out data/elliptic
graphshield-data audit   --data data/elliptic --train-end 29 --val-end 34

# Main comparison: tabular baselines, neighbour-feature ablation, MLP, GCN/SAGE/GAT, 5 seeds
graphshield-lab run --data data/elliptic --train-end 29 --val-end 34 \
    --models logreg rf hgb logreg+agg rf+agg hgb+agg mlp gcn sage gat \
    --seeds 0 1 2 3 4 --out "$OUT/main.json"

# Drift: static vs rolling retraining across the time-step-43 market shutdown
graphshield-lab drift --data data/elliptic --models hgb gcn sage \
    --window 12 --horizon 4 --static-train-end 29 --static-val-end 34 \
    --seeds 0 1 2 --out "$OUT/drift.json"
