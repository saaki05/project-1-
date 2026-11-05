# graphshield-model-lab

Models and evaluation protocol for the GraphShield study of **temporal robustness of GNNs for
illicit-transaction detection**. It consumes datasets produced by
`graphshield-data-engine` (sibling directory) and compares
graph neural networks against tabular baselines under one leak-aware protocol.

## What is implemented

| Component | Details |
|---|---|
| Graph layers | `GCNLayer`, `SAGELayer` (mean), `GATLayer` (multi-head) written on edge lists with scatter ops, no `torch_geometric` dependency |
| Layer verification | Each layer is tested against a dense-matrix reference and cross-checked numerically against the matching `torch_geometric` layer (GCN, GraphSAGE, GAT) when PyG is installed (`pip install -e ".[crosscheck]"`) |
| Baselines | Logistic regression, random forest, histogram gradient boosting; `+agg` variants add one-hop mean neighbour features to test how much of a GNN's gain is neighbourhood averaging |
| Neural feature-only baseline | `mlp` |
| Metrics | PR-AUC, ROC-AUC, F1 and MCC at a validation-chosen threshold, precision@k, ECE, per-time-step metrics, bootstrap CIs, paired bootstrap difference tests |
| Harness | Multi-seed runs, leakage-audit gate (refuses to run on a split that fails `graphshield-data audit`), JSON output |
| Drift study | `graphshield-lab drift`: static training vs rolling-window retraining, evaluated block by block on identical test nodes (every training split audited) |

## Evaluation protocol

1. Training sees only the subgraph of nodes up to the last training step.
2. Feature standardisation uses training-split statistics only.
3. Early stopping (validation PR-AUC) and the F1 threshold use the validation split only.
4. Test metrics are computed once with those choices fixed.
5. Results are reported as mean ± std over seeds, with a bootstrap CI for the seed-averaged
   predictions.

## Install

```bash
git clone https://github.com/saaki05/project-1- && cd project-1-
pip install -e graphshield-data-engine
cd graphshield-model-lab
pip install -e ".[dev]"
pytest
```

## Run

```bash
graphshield-data convert --raw /path/to/elliptic --out data/elliptic   # from the data engine
graphshield-lab run --data data/elliptic --train-end 29 --val-end 34 \
    --models logreg rf hgb hgb+agg gcn sage gat --seeds 0 1 2 3 4 --out results/main.json
graphshield-lab drift --data data/elliptic --models hgb gcn sage --window 12 --horizon 4 \
    --static-train-end 29 --static-val-end 34 --out results/drift.json
graphshield-lab models      # list available model names
```

## Results

This repository ships **no benchmark numbers**. Real-data results will be added to
`results/` and the table below only after they have been produced and checked; the unit tests
use synthetic graphs purely to verify code paths and say nothing about real fraud detection.

| Experiment | Status |
|---|---|
| Elliptic, temporal split (train 1–29, val 30–34, test 35–49) | not yet run |
| Drift mitigation (rolling windows) | implemented and tested on synthetic graphs only; not yet run on Elliptic |
| Ablations (depth, aggregation, undirected edges) | not yet run |

## Limitations

- Full-batch training only; very large graphs would need neighbour sampling.
- The paired bootstrap resamples evaluation nodes; it does not account for seed variance, so
  report it together with the multi-seed spread.
- Hyperparameter search (Optuna) is declared as an optional extra but not yet wired in.

## License

MIT.
