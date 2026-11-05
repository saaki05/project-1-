#!/usr/bin/env bash
# Tuned study: 8 random-search trials per model family on a reserved 20% tuning pool,
# then 10 fresh splits of the remaining labelled nodes.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
graphshield-lab static --amazon "${AMAZON_MAT:?path to Amazon.mat}" --relation upu \
  --models logreg rf hgb logreg+agg rf+agg hgb+agg mlp gcn sage gat \
  --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 100 --tune-trials 8 --out static_upu_tuned.json > REPORT_upu_tuned.md
