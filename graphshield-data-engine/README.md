# graphshield-data-engine

Data layer for the GraphShield research project on **temporal robustness of graph neural
networks for illicit-transaction detection**. It owns everything that happens before a model
sees data:

- a typed in-memory graph (`GraphData`) with validation and induced-subgraph extraction;
- a reader/writer for the raw Elliptic Bitcoin CSV layout;
- a versioned Parquet store with a SHA-256 manifest (the contract with downstream repos);
- strict temporal splits and rolling windows for drift-aware retraining;
- a **split auditor** that detects overlap, look-ahead, unlabeled evaluation nodes,
  train/test edges (transductive leakage) and duplicated feature rows;
- a loader for the Amazon fraud graph (`load_amazon`, all four relations; the 3,305
  placeholder-labelled nodes are marked unknown) and `stratified_split` for graphs without
  time information (audited with ordering checks switched off and a note that evaluation is
  transductive);
- a synthetic generator used only for tests and smoke runs.

It has no deep-learning dependency (NumPy, pandas, pyarrow only).

## Install

```bash
git clone https://github.com/saaki05/project-1-
cd project-1-/graphshield-data-engine
pip install -e ".[dev]"
pytest
```

## Usage

Obtain the Elliptic data yourself from its official source and read its license terms
first. This repository does not redistribute it.

```bash
graphshield-data convert --raw /path/to/elliptic_bitcoin_dataset --out data/elliptic
graphshield-data audit   --data data/elliptic --train-end 29 --val-end 34
```

```python
from graphshield_data import temporal_split, audit_split
from graphshield_data.store import load_graph

g = load_graph("data/elliptic")
split = temporal_split(g, train=(1, 29), val=(30, 34), test=(35, 49))
assert audit_split(g, split).ok
train_graph, kept = g.subgraph(split.train)   # train without test-time edges
```

## Interface contract

`save_graph` writes `nodes.parquet` (`node_id, timestep, label, f0..fF`), `edges.parquet`
(`src, dst` as external ids) and `manifest.json` (`schema_version`, counts, checksums).
`load_graph` refuses a different `schema_version` and verifies checksums.
Labels: `-1` unknown, `0` licit, `1` illicit.

## Design notes

- Splits only ever select *labelled* nodes, so evaluation counts never include unknowns.
- The auditor reports cross-split edges as a warning, because training a GNN on the full graph
  lets test-time structure influence training. Use `GraphData.subgraph` to avoid it.
- Synthetic data has class-dependent feature shifts and an optional mid-sequence concept
  shift. It exists to exercise code paths; no scientific claim may rest on it.

## Status

`v0.1.0`: Elliptic layout and Amazon fraud graph (`pip install -e ".[amazon]"` for scipy).
Planned: Elliptic++ and synthetic-AML loaders.

## License

MIT. The Elliptic dataset is covered by its own license; check it before use.
