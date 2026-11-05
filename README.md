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

![GraphShield architecture: data engine, model lab, intelligence service](docs/figures/architecture.png)

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
| Experiments on the real Elliptic dataset (temporal) | **not run**: the download was unreachable from the development container |
| Experiments on a real fraud graph (Amazon, static) | **run**, 10 seeds; see Results on real data |
| Static-vs-rolling drift study (`graphshield-lab drift`) | implemented; tested on synthetic graphs with a built-in concept shift (rolling retraining wins there, as expected, which only shows the code works) |
| Hyperparameter search, ablations, statistical comparison tables on real data | **not run**; `scripts/run_elliptic.sh` reproduces the planned runs once the data is available |
| Docker image | Dockerfile written, never built |

No benchmark numbers appear in this repository because none have been measured on real data.
Smoke runs on synthetic graphs only verify that the code executes. The Elliptic download was
not reachable from the development container (Kaggle and the PyG mirror were blocked), and
the container has no GPU.

## Results on real data

### Amazon fraud graph (static, no timestamps)

Real user-review fraud graph (11,944 users; 8,639 labelled, 9.5% fraud; 25 handcrafted
features; the first 3,305 placeholder nodes are excluded as unlabelled). Fresh stratified
40/20/40 split per seed, transductive training, 10 seeds, user–product–user (UPU) relation
(351,216 directed edges). Full tables and per-seed values:
[`graphshield-model-lab/results/amazon/`](graphshield-model-lab/results/amazon). Mean ± std:

| model | PR-AUC | ROC-AUC | F1 | P@100 |
|---|---|---|---|---|
| logistic regression | 0.884 ± 0.013 | 0.970 ± 0.005 | 0.839 ± 0.017 | 0.965 ± 0.014 |
| random forest | 0.898 ± 0.009 | 0.973 ± 0.006 | 0.847 ± 0.013 | 0.976 ± 0.010 |
| gradient boosting (hgb) | 0.899 ± 0.008 | 0.973 ± 0.006 | 0.854 ± 0.011 | 0.977 ± 0.009 |
| hgb + neighbour-mean features | **0.916 ± 0.012** | 0.977 ± 0.006 | **0.863 ± 0.014** | 0.993 ± 0.008 |
| MLP | 0.867 ± 0.015 | 0.964 ± 0.005 | 0.837 ± 0.019 | 0.970 ± 0.013 |
| GraphSAGE | 0.862 ± 0.016 | 0.966 ± 0.007 | 0.818 ± 0.019 | 0.962 ± 0.014 |
| GCN | 0.552 ± 0.017 | 0.875 ± 0.007 | 0.520 ± 0.010 | 0.820 ± 0.036 |
| GAT | 0.596 ± 0.063 | 0.893 ± 0.012 | 0.573 ± 0.067 | 0.843 ± 0.036 |

![PR-AUC of every model over ten random splits](docs/figures/model_comparison.png)

Paired by seed (same split for every model), PR-AUC differences with bootstrap 95% CIs:

![Paired PR-AUC differences to plain gradient boosting](docs/figures/paired_contrasts.png)

- gradient boosting with neighbour-averaged features beats plain gradient boosting by +0.017
  [+0.013, +0.021] (10/10 seeds) and beats GraphSAGE by 0.054 [0.047, 0.063] (10/10 seeds);
- GraphSAGE is 0.037 [0.030, 0.045] below plain gradient boosting (0/10 seeds better);
- GCN and GAT are far below every other model (−0.346 and −0.303 against gradient boosting).

The denser UVU relation (2.07 M edges, 3 seeds, 60 epochs) gives the same ordering: gradient
boosting + neighbour features 0.917 ± 0.008, plain gradient boosting 0.905 ± 0.005, GraphSAGE
0.862 ± 0.008, GCN 0.574 ± 0.025.

![Share of a fraud user's neighbours that are fraud](docs/figures/neighbourhood.png)

**Reading these results.** On this graph an untuned tree ensemble matches or beats
every GNN, and most of what graph structure contributes is captured by simply averaging
neighbours' features. GCN and GAT mix a node's features with its neighbours' without a
separate self-weight; here only 17.7% of a fraud node's UPU neighbours are themselves fraud
(base rate 9.5%) and edge homophily (0.785) is below the 0.828 expected under random
wiring, so neighbourhood averaging dilutes the fraud signal. GraphSAGE keeps a separate self
term and is far better. That explanation fits the measurements but was not tested by an
intervention.

**Caveats.** All models use default or lightly set hyper-parameters; the GNNs were not tuned
(2 layers, 64 hidden units, 100 epochs, learning rate 5e-3), so the GCN/GAT gap may shrink
with tuning. One relation per run. Random, non-temporal splits with transductive training
(test-node features and edges are visible during training, labels are not). This graph says
nothing about temporal drift; the Elliptic study below is still pending.

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

## Regenerating the figures

Every figure in `docs/figures/` is produced from the result files by a script (no number is
typed by hand). `pip install matplotlib`, then:

```bash
python scripts/amazon_homophily.py /path/to/Amazon.mat graphshield-model-lab/results/amazon/homophily.json
python scripts/make_figures.py
```

## License

MIT. Third-party datasets keep their own licenses.
