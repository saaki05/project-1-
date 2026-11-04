"""In-memory representation of a temporal transaction graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = -1
LICIT = 0
ILLICIT = 1


@dataclass
class GraphData:
    """A directed graph whose nodes carry a time step, features and an optional label.

    Attributes:
        node_ids: external identifiers, shape (N,), unique.
        timestep: integer time step of each node, shape (N,).
        features: float32 matrix, shape (N, F).
        labels: int8 vector, shape (N,): -1 unknown, 0 licit, 1 illicit.
        edge_index: int64 array of shape (2, E) holding *row indices* into the node arrays
            (not external ids).
    """

    node_ids: np.ndarray
    timestep: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    edge_index: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    def validate(self) -> None:
        """Raise ``ValueError`` if the structure is internally inconsistent."""
        n = self.num_nodes
        if self.timestep.shape != (n,) or self.labels.shape != (n,):
            raise ValueError("timestep and labels must have shape (N,)")
        if self.features.ndim != 2 or self.features.shape[0] != n:
            raise ValueError("features must have shape (N, F)")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if len(np.unique(self.node_ids)) != n:
            raise ValueError("node_ids must be unique")
        if not np.isfinite(self.features).all():
            raise ValueError("features contain NaN or infinite values")
        if not np.isin(self.labels, (UNKNOWN, LICIT, ILLICIT)).all():
            raise ValueError("labels must be in {-1, 0, 1}")
        if self.num_edges and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge_index refers to a node outside [0, N)")

    def subgraph(self, node_mask: np.ndarray) -> tuple[GraphData, np.ndarray]:
        """Return the induced subgraph on ``node_mask`` and the kept row indices.

        Only edges with both endpoints selected are retained, and indices are remapped.
        """
        node_mask = np.asarray(node_mask, dtype=bool)
        if node_mask.shape != (self.num_nodes,):
            raise ValueError("node_mask must have shape (N,)")
        kept = np.flatnonzero(node_mask)
        remap = np.full(self.num_nodes, -1, dtype=np.int64)
        remap[kept] = np.arange(kept.size)
        src, dst = self.edge_index
        keep_edge = node_mask[src] & node_mask[dst]
        edge_index = np.stack([remap[src[keep_edge]], remap[dst[keep_edge]]])
        sub = GraphData(
            node_ids=self.node_ids[kept],
            timestep=self.timestep[kept],
            features=self.features[kept],
            labels=self.labels[kept],
            edge_index=edge_index,
        )
        return sub, kept
