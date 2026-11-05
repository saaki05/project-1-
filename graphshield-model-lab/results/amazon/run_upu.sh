#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
graphshield-lab static --amazon "${AMAZON_MAT:?path to Amazon.mat}" --relation upu \
  --models logreg rf hgb logreg+agg rf+agg hgb+agg mlp gcn sage gat \
  --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 100 --out static_upu.json > static_upu.md
