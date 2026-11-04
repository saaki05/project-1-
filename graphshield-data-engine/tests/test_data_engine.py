import numpy as np
import pytest

from graphshield_data import GraphData, audit_split, rolling_windows, temporal_split
from graphshield_data.cli import main
from graphshield_data.elliptic import load_elliptic, write_elliptic
from graphshield_data.store import load_graph, save_graph
from graphshield_data.synthetic import make_synthetic_graph


@pytest.fixture(scope="module")
def graph():
    return make_synthetic_graph(num_steps=12, nodes_per_step=100, num_features=8, seed=3)


def test_synthetic_graph_is_valid_and_reproducible(graph):
    graph.validate()
    again = make_synthetic_graph(num_steps=12, nodes_per_step=100, num_features=8, seed=3)
    assert np.array_equal(graph.features, again.features)
    assert np.array_equal(graph.edge_index, again.edge_index)
    # edges never cross time steps
    src, dst = graph.edge_index
    assert (graph.timestep[src] == graph.timestep[dst]).all()


def test_validate_rejects_bad_edges(graph):
    bad = GraphData(graph.node_ids, graph.timestep, graph.features, graph.labels,
                    np.array([[0], [graph.num_nodes]]))
    with pytest.raises(ValueError):
        bad.validate()


def test_subgraph_remaps_edges(graph):
    mask = graph.timestep <= 3
    sub, kept = graph.subgraph(mask)
    assert sub.num_nodes == mask.sum()
    assert sub.edge_index.max() < sub.num_nodes
    assert np.array_equal(sub.node_ids, graph.node_ids[kept])
    # every retained edge existed in the original between the same ids
    orig = set(zip(graph.node_ids[graph.edge_index[0]], graph.node_ids[graph.edge_index[1]],
                   strict=True))
    for s, d in zip(sub.node_ids[sub.edge_index[0]], sub.node_ids[sub.edge_index[1]], strict=True):
        assert (s, d) in orig


def test_elliptic_csv_round_trip(graph, tmp_path):
    write_elliptic(graph, tmp_path / "raw")
    back = load_elliptic(tmp_path / "raw")
    assert np.array_equal(back.node_ids, graph.node_ids)
    assert np.array_equal(back.labels, graph.labels)
    assert np.array_equal(back.edge_index, graph.edge_index)
    assert np.allclose(back.features, graph.features, atol=1e-5)


def test_store_round_trip_and_checksum(graph, tmp_path):
    save_graph(graph, tmp_path)
    back = load_graph(tmp_path)
    assert np.array_equal(back.edge_index, graph.edge_index)
    assert np.array_equal(back.timestep, graph.timestep)
    (tmp_path / "nodes.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_graph(tmp_path)


def test_temporal_split_masks_labelled_only(graph):
    s = temporal_split(graph, (1, 6), (7, 8), (9, 12))
    for m in (s.train, s.val, s.test):
        assert (graph.labels[m] >= 0).all()
    assert not (s.train & s.test).any()
    with pytest.raises(ValueError):
        temporal_split(graph, (1, 8), (7, 9), (10, 12))


def test_audit_passes_clean_split_and_flags_leaks(graph):
    s = temporal_split(graph, (1, 6), (7, 8), (9, 12))
    rep = audit_split(graph, s)
    assert rep.ok
    codes = {f.code for f in rep.findings}
    assert "cross-split-edges" in codes

    # force an overlap: audit must fail
    leaky = type(s)(s.name, s.train, s.val, s.test | s.train)
    rep = audit_split(graph, leaky)
    assert not rep.ok
    assert any(f.code == "overlap" for f in rep.findings)
    assert any(f.code == "temporal-order" for f in rep.findings)


def test_rolling_windows_never_look_ahead(graph):
    windows = list(rolling_windows(graph, window=5, horizon=2, val_steps=1))
    assert windows
    for w in windows:
        assert graph.timestep[w.train].max() < graph.timestep[w.val].min()
        assert graph.timestep[w.val].max() < graph.timestep[w.test].min()
        assert audit_split(graph, w).ok


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "data"
    assert main(["synth", "--out", str(out), "--steps", "10", "--nodes-per-step", "80"]) == 0
    assert main(["audit", "--data", str(out), "--train-end", "6", "--val-end", "8"]) == 0
    assert "RESULT: OK" in capsys.readouterr().out
