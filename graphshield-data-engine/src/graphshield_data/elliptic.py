"""Reader and writer for the Elliptic Bitcoin dataset's raw CSV layout.

Raw files (obtain them from the dataset's official distribution and check its license):

* ``elliptic_txs_features.csv``: no header; column 0 is the transaction id, column 1 the time
  step (1..49) and the remaining 165 columns are features.
* ``elliptic_txs_classes.csv``: header ``txId,class`` with class in {"unknown", "1", "2"};
  "1" is illicit and "2" is licit.
* ``elliptic_txs_edgelist.csv``: header ``txId1,txId2``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graphshield_data.graph import ILLICIT, LICIT, UNKNOWN, GraphData

FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"

_CLASS_TO_LABEL = {"unknown": UNKNOWN, "1": ILLICIT, "2": LICIT}
_LABEL_TO_CLASS = {v: k for k, v in _CLASS_TO_LABEL.items()}


def load_elliptic(raw_dir: str | Path) -> GraphData:
    """Load the raw CSV files into a validated :class:`GraphData`."""
    raw = Path(raw_dir)
    feats = pd.read_csv(raw / FEATURES_FILE, header=None)
    classes = pd.read_csv(raw / CLASSES_FILE, dtype={"class": str})
    edges = pd.read_csv(raw / EDGES_FILE)

    node_ids = feats.iloc[:, 0].to_numpy(dtype=np.int64)
    timestep = feats.iloc[:, 1].to_numpy(dtype=np.int16)
    features = feats.iloc[:, 2:].to_numpy(dtype=np.float32)

    unexpected = set(classes["class"].unique()) - set(_CLASS_TO_LABEL)
    if unexpected:
        raise ValueError(f"unexpected class values: {sorted(unexpected)}")
    label_by_id = classes.set_index("txId")["class"].map(_CLASS_TO_LABEL)
    labels = label_by_id.reindex(node_ids).to_numpy()
    if np.isnan(labels.astype(float)).any():
        raise ValueError("some transactions have no entry in the classes file")
    labels = labels.astype(np.int8)

    index_of = pd.Series(np.arange(node_ids.size), index=node_ids)
    src = index_of.reindex(edges["txId1"].to_numpy())
    dst = index_of.reindex(edges["txId2"].to_numpy())
    if src.isna().any() or dst.isna().any():
        raise ValueError("edge list refers to transactions missing from the features file")
    edge_index = np.stack([src.to_numpy(dtype=np.int64), dst.to_numpy(dtype=np.int64)])

    graph = GraphData(node_ids, timestep, features, labels, edge_index)
    graph.validate()
    return graph


def write_elliptic(graph: GraphData, raw_dir: str | Path) -> None:
    """Write ``graph`` in the raw Elliptic CSV layout (used for round-trip tests)."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    table = np.column_stack([graph.node_ids, graph.timestep, graph.features])
    pd.DataFrame(table).to_csv(raw / FEATURES_FILE, header=False, index=False)
    pd.DataFrame(
        {
            "txId": graph.node_ids,
            "class": [_LABEL_TO_CLASS[int(v)] for v in graph.labels],
        }
    ).to_csv(raw / CLASSES_FILE, index=False)
    pd.DataFrame(
        {
            "txId1": graph.node_ids[graph.edge_index[0]],
            "txId2": graph.node_ids[graph.edge_index[1]],
        }
    ).to_csv(raw / EDGES_FILE, index=False)
