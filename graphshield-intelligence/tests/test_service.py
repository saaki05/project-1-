import numpy as np
import pytest
from fastapi.testclient import TestClient
from graphshield_data import temporal_split
from graphshield_data.synthetic import make_synthetic_graph
from graphshield_lab.runner import TrainConfig, run_model

from graphshield_service.api import create_app
from graphshield_service.bundle import load_bundle, save_bundle
from graphshield_service.drift import drift_report, ks_statistic, psi


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    g = make_synthetic_graph(num_steps=12, nodes_per_step=120, num_features=8,
                             labelled_fraction=0.9, seed=2)
    split = temporal_split(g, (1, 7), (8, 9), (10, 12))
    pred = run_model("gcn", g, split, 0, TrainConfig(epochs=30, patience=30))
    path = tmp_path_factory.mktemp("bundle") / "model.pt"
    save_bundle(pred.artifact, path)
    return g, path, pred


def test_bundle_round_trip_matches_training_probabilities(trained):
    g, path, pred = trained
    bundle = load_bundle(path)
    mask = g.timestep <= 12
    sub, kept = g.subgraph(mask)
    probs = bundle.score(sub.features, sub.edge_index)
    test_idx = np.searchsorted(kept, np.flatnonzero(temporal_split(
        g, (1, 7), (8, 9), (10, 12)).test))
    assert np.allclose(probs[test_idx], pred.test_prob, atol=1e-4)


def test_saliency_shapes(trained):
    g, path, _ = trained
    bundle = load_bundle(path)
    feat, node = bundle.saliency(g.features[:50], np.zeros((2, 0), dtype=np.int64), 3)
    assert feat.shape == (8,) and node.shape == (50,)
    # with no edges only the node itself can influence its score
    assert node[3] > 0 and np.count_nonzero(node) == 1


def test_api_endpoints(trained):
    g, path, _ = trained
    client = TestClient(create_app(path))
    assert client.get("/health").json()["status"] == "ok"
    body = {"features": g.features[:20].tolist(), "edges": [[0, 1], [1, 2]]}
    r = client.post("/score", json=body)
    assert r.status_code == 200 and len(r.json()["probabilities"]) == 20
    r = client.post("/explain", json={**body, "node": 1})
    assert r.status_code == 200 and len(r.json()["top_features"]) > 0


def test_api_rejects_bad_input(trained):
    _, path, _ = trained
    client = TestClient(create_app(path))
    assert client.post("/score", json={"features": [[1.0, 2.0]]}).status_code == 422
    good = {"features": [[0.0] * 8, [0.0] * 8], "edges": [[0, 5]]}
    assert client.post("/score", json=good).status_code == 422
    assert client.post("/explain", json={**good, "edges": [], "node": 9}).status_code == 422


def test_drift_statistics():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.05
    assert psi(a, rng.normal(loc=1.5, size=5000)) > 0.25
    assert ks_statistic(a, a) == 0.0
    assert 0.5 < ks_statistic(a, a + 3) <= 1.0
    ref = rng.normal(size=(2000, 4))
    cur = ref.copy() + np.array([0, 0, 2.0, 0]) + rng.normal(scale=0.01, size=ref.shape)
    rep = drift_report(ref, cur, top=2)
    assert rep["top"][0]["feature"] == 2
