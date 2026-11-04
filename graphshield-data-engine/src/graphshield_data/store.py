"""Versioned Parquet storage with a checksum manifest.

The manifest is the contract between this package and downstream repositories: consumers
read ``manifest.json`` first and must refuse data whose ``schema_version`` they do not know.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graphshield_data.graph import GraphData

SCHEMA_VERSION = 1
NODES_FILE = "nodes.parquet"
EDGES_FILE = "edges.parquet"
MANIFEST_FILE = "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_graph(graph: GraphData, out_dir: str | Path) -> dict:
    """Write nodes, edges and manifest; return the manifest dictionary."""
    graph.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = {f"f{i}": graph.features[:, i] for i in range(graph.num_features)}
    nodes = pd.DataFrame({"node_id": graph.node_ids, "timestep": graph.timestep,
                          "label": graph.labels, **cols})
    nodes.to_parquet(out / NODES_FILE, index=False)
    pd.DataFrame(
        {"src": graph.node_ids[graph.edge_index[0]], "dst": graph.node_ids[graph.edge_index[1]]}
    ).to_parquet(out / EDGES_FILE, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_features": graph.num_features,
        "files": {name: _sha256(out / name) for name in (NODES_FILE, EDGES_FILE)},
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_graph(data_dir: str | Path, verify: bool = True) -> GraphData:
    """Load a graph written by :func:`save_graph`, verifying checksums by default."""
    d = Path(data_dir)
    manifest = json.loads((d / MANIFEST_FILE).read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {manifest['schema_version']} (expected {SCHEMA_VERSION})"
        )
    if verify:
        for name, digest in manifest["files"].items():
            if _sha256(d / name) != digest:
                raise ValueError(f"checksum mismatch for {name}")

    nodes = pd.read_parquet(d / NODES_FILE)
    edges = pd.read_parquet(d / EDGES_FILE)
    feat_cols = [f"f{i}" for i in range(manifest["num_features"])]
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    index_of = pd.Series(np.arange(node_ids.size), index=node_ids)
    edge_index = np.stack(
        [
            index_of.reindex(edges["src"].to_numpy()).to_numpy(dtype=np.int64),
            index_of.reindex(edges["dst"].to_numpy()).to_numpy(dtype=np.int64),
        ]
    )
    graph = GraphData(
        node_ids=node_ids,
        timestep=nodes["timestep"].to_numpy(dtype=np.int16),
        features=nodes[feat_cols].to_numpy(dtype=np.float32),
        labels=nodes["label"].to_numpy(dtype=np.int8),
        edge_index=edge_index,
    )
    graph.validate()
    return graph
