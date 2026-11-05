"""Loader for the Amazon fraud graph (``Amazon.mat``) used in the CARE-GNN line of work.

Users are nodes, with 25 handcrafted behaviour features and a binary "fraud" label. Three
relations are provided (``upu``: reviewed the same product, ``usu``: same star rating within a
week, ``uvu``: top-5% mutual review-text similarity) plus their union (``homo``).

Important quirk handled here: the first 3,305 nodes carry placeholder label 0 and are
unlabelled in the original protocol, so they are marked unknown rather than licit.

The graph has no timestamps (every node gets ``timestep = 1``), so it supports the static
GNN-versus-tabular question but not drift analysis. Provenance: Zhang et al., "Spam Review
Detection with Graph Convolutional Networks" (CIKM 2019) and Dou et al., "Enhancing Graph
Neural Network-based Fraud Detectors against Camouflaged Fraudsters" (CIKM 2020). Check the
terms of the original data source before redistributing; the file is not included here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from graphshield_data.graph import UNKNOWN, GraphData

RELATIONS = ("homo", "upu", "usu", "uvu")
UNLABELLED_PREFIX = 3305


def load_amazon(path: str | Path, relation: str = "upu",
                unlabelled_prefix: int = UNLABELLED_PREFIX) -> GraphData:
    """Load ``Amazon.mat`` into a validated :class:`GraphData` for one relation."""
    import scipy.io as sio  # imported lazily: only needed for this loader

    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}")
    key = "homo" if relation == "homo" else f"net_{relation}"
    mat = sio.loadmat(str(path), squeeze_me=False)
    features = np.asarray(mat["features"].todense() if hasattr(mat["features"], "todense")
                          else mat["features"], dtype=np.float32)
    labels = np.asarray(mat["label"]).ravel().astype(np.int8)
    labels[:unlabelled_prefix] = UNKNOWN
    adj = mat[key].tocoo()
    keep = adj.row != adj.col  # drop any self loops
    edge_index = np.stack([adj.row[keep], adj.col[keep]]).astype(np.int64)
    n = features.shape[0]
    graph = GraphData(node_ids=np.arange(n, dtype=np.int64), timestep=np.ones(n, dtype=np.int16),
                      features=features, labels=labels, edge_index=edge_index)
    graph.validate()
    return graph
