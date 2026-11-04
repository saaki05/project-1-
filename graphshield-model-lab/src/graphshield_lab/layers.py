"""Message-passing layers written directly on edge lists (no torch_geometric).

Convention: ``edge_index[0]`` is the source and ``edge_index[1]`` the destination of a
message; every layer aggregates over *incoming* edges at the destination node.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loops, loops])], dim=1)


def to_undirected(edge_index: Tensor) -> Tensor:
    """Add reverse edges and drop duplicates."""
    both = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(both, dim=1)


def scatter_add(src: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    out = src.new_zeros((num_nodes, *src.shape[1:]))
    return out.index_add_(0, index, src)


def scatter_softmax(score: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Softmax of ``score`` (E, H) over the edges sharing a destination index."""
    heads = score.shape[1]
    idx = index.unsqueeze(1).expand(-1, heads)
    maxima = score.new_full((num_nodes, heads), float("-inf"))
    maxima = maxima.scatter_reduce(0, idx, score, reduce="amax", include_self=True)
    exp = (score - maxima[index]).exp()
    denom = scatter_add(exp, index, num_nodes)
    return exp / denom[index].clamp_min(1e-16)


class GCNLayer(nn.Module):
    """Kipf & Welling graph convolution: D^-1/2 (A + I) D^-1/2 X W with in-degree D."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.shape[0]
        ei = add_self_loops(edge_index, n)
        src, dst = ei
        deg = scatter_add(torch.ones(ei.shape[1], device=x.device, dtype=x.dtype), dst, n)
        dinv = deg.pow(-0.5)
        norm = dinv[src] * dinv[dst]
        h = self.lin(x)
        return scatter_add(h[src] * norm.unsqueeze(1), dst, n) + self.bias


class SAGELayer(nn.Module):
    """GraphSAGE with a mean aggregator: W_self x_v + W_neigh mean_{u -> v} x_u."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.shape[0]
        src, dst = edge_index
        total = scatter_add(x[src], dst, n)
        deg = scatter_add(torch.ones(src.shape[0], device=x.device, dtype=x.dtype), dst, n)
        mean = total / deg.clamp_min(1.0).unsqueeze(1)
        return self.lin_self(x) + self.lin_neigh(mean)


class GATLayer(nn.Module):
    """Graph attention (Velickovic et al.) with multi-head concatenation or averaging."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.negative_slope = negative_slope
        self.attn_dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        bound = math.sqrt(6.0 / (heads * out_dim + out_dim))
        nn.init.uniform_(self.att_src, -bound, bound)
        nn.init.uniform_(self.att_dst, -bound, bound)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.shape[0]
        src, dst = add_self_loops(edge_index, n)
        h = self.lin(x).view(n, self.heads, self.out_dim)
        e_src = (h * self.att_src).sum(-1)
        e_dst = (h * self.att_dst).sum(-1)
        score = nn.functional.leaky_relu(e_src[src] + e_dst[dst], self.negative_slope)
        alpha = self.attn_dropout(scatter_softmax(score, dst, n))
        out = scatter_add(h[src] * alpha.unsqueeze(-1), dst, n)
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias
