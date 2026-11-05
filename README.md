# GraphShield

Research engineering project on **temporal robustness of graph neural networks for
illicit-transaction detection**. It is organised as three independently installable packages
with one-way dependencies:

```
graphshield-data-engine  ──▶  graphshield-model-lab  ──▶  graphshield-intelligence
 datasets, splits, audit       GNN layers, baselines,        scoring / explanation /
 (numpy, pandas, pyarrow)      metrics, protocol (torch)     drift service (FastAPI)
```

| Package | Responsibility | Contract it publishes |
|---|---|---|
| [`graphshield-data-engine`](graphshield-data-engine) | load, validate, store and split temporal graphs; audit splits for leakage | Parquet + `manifest.json` (schema v1) |
| [`graphshield-model-lab`](graphshield-model-lab) | from-scratch GCN/GraphSAGE/GAT, tabular baselines, metrics, multi-seed protocol | model bundle (format v1) |
| [`graphshield-intelligence`](graphshield-intelligence) | HTTP scoring, saliency explanations, PSI/KS drift reports | REST API |

Each package has its own `pyproject.toml`, tests and README.

## Research question

On the Elliptic Bitcoin transaction graph, published GNN results are hard to compare because
splits, seeds and leakage controls differ. Under a strict, audited temporal protocol:

1. Is the GNN-versus-gradient-boosting gap real, or explained by neighbourhood-averaged
   features and evaluation choices?
2. Which drift mitigations (rolling retraining windows, feature pruning) help after the
   dark-market shutdown, when all models are known to degrade?
3. How do models compare on analyst-facing metrics (precision@k, calibration), not just F1?

The intended contribution is a leak-audited, multi-seed benchmark with ablations, not a new
architecture.

## Status (honest summary)

| Item | State |
|---|---|
| Data engine, GNN layers, baselines, metrics, protocol, service | implemented; unit tests pass |
| Layers verified against dense reference computations | yes |
| Cross-check against PyTorch Geometric | GCN, GraphSAGE and GAT layers match `torch_geometric` outputs to 1e-5 (tests run when PyG is installed; verified in development) |
| Experiments on the real Elliptic dataset | **not run**: the data must be downloaded under its own license and the work needs your Kaggle/Colab account |
| Static-vs-rolling drift study (`graphshield-lab drift`) | implemented; tested on synthetic graphs with a built-in concept shift (rolling retraining wins there, as expected, which only shows the code works) |
| Hyperparameter search, ablations, statistical comparison tables on real data | **not run**; `scripts/run_elliptic.sh` reproduces the planned runs once the data is available |
| Docker image | Dockerfile written, never built |

No benchmark numbers appear in this repository because none have been measured on real data.
Smoke runs on synthetic graphs only verify that the code executes. The Elliptic download was
not reachable from the development container (Kaggle and the PyG mirror were blocked), and
the container has no GPU.

## Quick start

```bash
pip install -e graphshield-data-engine -e graphshield-model-lab -e graphshield-intelligence
graphshield-data synth --out /tmp/synth --steps 30 --shift-step 22   # synthetic, tests only
graphshield-lab run --data /tmp/synth --train-end 15 --val-end 20 --models logreg gcn sage gat
```

For the real dataset, download Elliptic from its official source (check the license), then
`graphshield-data convert --raw <dir> --out data/elliptic`.

## Development

```bash
pip install -e "graphshield-data-engine[dev]" -e "graphshield-model-lab[dev]" -e "graphshield-intelligence[dev]"
for d in graphshield-*; do (cd $d && pytest -q); done
```

CI (`.github/workflows/ci.yml`) installs the three packages together and runs lint and tests.

## License

MIT. Third-party datasets keep their own licenses.
