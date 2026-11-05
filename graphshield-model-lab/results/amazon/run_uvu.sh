#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
graphshield-lab static --amazon "${AMAZON_MAT:?path to Amazon.mat}" --relation uvu \
  --models hgb hgb+agg rf+agg sage gcn --seeds 0 1 2 --epochs 60 --out static_uvu.json > static_uvu.md
