"""Node classifiers built from the layers in :mod:`graphshield_lab.layers`."""

from __future__ import annotations

from torch import Tensor, nn

from graphshield_lab.layers import GATLayer, GCNLayer, SAGELayer

GNN_KINDS = ("gcn", "sage", "gat")


class NodeClassifier(nn.Module):
    """Stack of graph layers followed by a linear head that outputs one logit per node.

    ``kind="mlp"`` ignores the graph and serves as the feature-only neural baseline.
    """

    def __init__(
        self,
        kind: str,
        in_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if kind not in (*GNN_KINDS, "mlp"):
            raise ValueError(f"unknown model kind {kind!r}")
        if kind == "gat" and hidden % heads:
            raise ValueError("hidden must be divisible by heads for GAT")
        self.kind = kind
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU() if kind == "gat" else nn.ReLU()

        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers):
            if kind == "gcn":
                layers.append(GCNLayer(dim, hidden))
            elif kind == "sage":
                layers.append(SAGELayer(dim, hidden))
            elif kind == "gat":
                layers.append(GATLayer(dim, hidden // heads, heads=heads, dropout=dropout))
            else:
                layers.append(nn.Linear(dim, hidden))
            dim = hidden
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x) if self.kind == "mlp" else layer(x, edge_index)
            x = self.act(x)
        return self.head(self.dropout(x)).squeeze(-1)
