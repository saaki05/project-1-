import pytest
import torch

from graphshield_lab.layers import (
    GATLayer,
    GCNLayer,
    SAGELayer,
    scatter_softmax,
    to_undirected,
)

N = 7


@pytest.fixture
def graph():
    torch.manual_seed(0)
    edges = {(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (5, 1), (1, 5), (6, 2)}
    ei = torch.tensor(sorted(edges)).T.contiguous()  # unique directed edges
    x = torch.randn(N, 5)
    return x, ei


def dense_adj(ei, n):
    a = torch.zeros(n, n)
    a[ei[1], ei[0]] = 1.0  # row = destination
    return a


def test_gcn_matches_dense_formula(graph):
    x, ei = graph
    layer = GCNLayer(5, 4)
    a = dense_adj(ei, N) + torch.eye(N)
    dinv = a.sum(1).pow(-0.5)
    a_hat = dinv[:, None] * a * dinv[None, :]
    expected = a_hat @ layer.lin(x) + layer.bias
    assert torch.allclose(layer(x, ei), expected, atol=1e-5)


def test_sage_matches_dense_mean(graph):
    x, ei = graph
    layer = SAGELayer(5, 4)
    a = dense_adj(ei, N)
    mean = (a @ x) / a.sum(1, keepdim=True).clamp_min(1)
    expected = layer.lin_self(x) + layer.lin_neigh(mean)
    assert torch.allclose(layer(x, ei), expected, atol=1e-5)


def test_gat_matches_dense_attention(graph):
    x, ei = graph
    layer = GATLayer(5, 3, heads=2, concat=True).eval()
    h = layer.lin(x).view(N, 2, 3)
    a = dense_adj(ei, N) + torch.eye(N)
    expected = torch.zeros(N, 2, 3)
    for d in range(N):
        srcs = torch.nonzero(a[d]).flatten()
        e = (h[srcs] * layer.att_src).sum(-1) + (h[d] * layer.att_dst).sum(-1)
        w = torch.softmax(torch.nn.functional.leaky_relu(e, 0.2), dim=0)  # (S, H)
        expected[d] = (w.unsqueeze(-1) * h[srcs]).sum(0)
    expected = expected.reshape(N, -1) + layer.bias
    assert torch.allclose(layer(x, ei), expected, atol=1e-5)


def test_scatter_softmax_normalises_per_destination():
    score = torch.randn(6, 2)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    out = scatter_softmax(score, index, 3)
    for d in range(3):
        assert torch.allclose(out[index == d].sum(0), torch.ones(2), atol=1e-6)


def test_gradients_flow(graph):
    x, ei = graph
    for layer in (GCNLayer(5, 4), SAGELayer(5, 4), GATLayer(5, 2, heads=2)):
        layer(x, ei).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in layer.parameters())


def test_to_undirected_is_symmetric_and_unique(graph):
    _, ei = graph
    und = to_undirected(ei)
    pairs = {tuple(e) for e in und.T.tolist()}
    assert all((d, s) in pairs for s, d in pairs)
    assert und.shape[1] == len(pairs)


def test_matches_torch_geometric_when_available(graph):
    pyg = pytest.importorskip("torch_geometric.nn")
    x, ei = graph
    ours = GCNLayer(5, 4)
    theirs = pyg.GCNConv(5, 4)
    with torch.no_grad():
        theirs.lin.weight.copy_(ours.lin.weight)
        theirs.bias.copy_(ours.bias)
    assert torch.allclose(ours(x, ei), theirs(x, ei), atol=1e-5)


def test_sage_matches_torch_geometric_when_available(graph):
    pyg = pytest.importorskip("torch_geometric.nn")
    x, ei = graph
    ours = SAGELayer(5, 4)
    theirs = pyg.SAGEConv(5, 4, aggr="mean")
    with torch.no_grad():
        theirs.lin_l.weight.copy_(ours.lin_neigh.weight)
        theirs.lin_l.bias.copy_(ours.lin_self.bias)
        theirs.lin_r.weight.copy_(ours.lin_self.weight)
    assert torch.allclose(ours(x, ei), theirs(x, ei), atol=1e-5)


def test_gat_matches_torch_geometric_when_available(graph):
    pyg = pytest.importorskip("torch_geometric.nn")
    x, ei = graph
    ours = GATLayer(5, 3, heads=2, concat=True).eval()
    theirs = pyg.GATConv(5, 3, heads=2, concat=True, dropout=0.0).eval()
    with torch.no_grad():
        theirs.lin.weight.copy_(ours.lin.weight)
        theirs.att_src.copy_(ours.att_src)
        theirs.att_dst.copy_(ours.att_dst)
        theirs.bias.copy_(ours.bias)
    assert torch.allclose(ours(x, ei), theirs(x, ei), atol=1e-5)
