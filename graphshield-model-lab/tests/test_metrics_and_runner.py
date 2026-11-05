import numpy as np
import pytest
from graphshield_data import temporal_split
from graphshield_data.synthetic import make_synthetic_graph
from sklearn.metrics import average_precision_score

from graphshield_lab import metrics as M
from graphshield_lab.cli import run_experiment
from graphshield_lab.runner import TrainConfig, available_models, evaluate, run_model


def test_metrics_hand_computed():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
    assert M.precision_at_k(y, p, 2) == 0.5
    assert M.f1_at(y, p, 0.5) == pytest.approx(0.5)  # tp=1, fp=1, fn=1
    assert M.pr_auc(y, p) == pytest.approx(average_precision_score(y, p))
    assert M.ece(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert M.ece(np.array([0, 0]), np.array([0.95, 0.95])) == pytest.approx(0.95)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    lo, hi = M.bootstrap_ci(M.pr_auc, y, p, n_boot=300)
    assert lo <= M.pr_auc(y, p) <= hi


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good = np.clip(y * 0.5 + rng.random(600) * 0.5, 0, 1)
    noise = rng.random(600)
    res = M.paired_bootstrap_diff(M.pr_auc, y, good, noise, n_boot=300)
    assert res["diff"] > 0.1 and res["ci_low"] > 0 and res["p_value"] < 0.05
    same = M.paired_bootstrap_diff(M.pr_auc, y, good, good, n_boot=100)
    assert same["diff"] == 0.0


@pytest.fixture(scope="module")
def setup():
    g = make_synthetic_graph(num_steps=14, nodes_per_step=150, num_features=10,
                             labelled_fraction=0.8, seed=5)
    split = temporal_split(g, (1, 8), (9, 10), (11, 14))
    return g, split


@pytest.mark.parametrize("name", ["logreg", "hgb+agg", "mlp", "gcn", "sage", "gat"])
def test_models_beat_chance_on_synthetic(setup, name):
    g, split = setup
    pred = run_model(name, g, split, seed=0, cfg=TrainConfig(epochs=60, patience=15))
    y = g.labels[split.test]
    assert M.pr_auc(y, pred.test_prob) > 2 * y.mean()


def test_training_is_reproducible(setup):
    g, split = setup
    cfg = TrainConfig(epochs=25, patience=25)
    a = run_model("gcn", g, split, seed=3, cfg=cfg)
    b = run_model("gcn", g, split, seed=3, cfg=cfg)
    assert np.allclose(a.test_prob, b.test_prob)


def test_run_experiment_summary_and_audit_gate(setup):
    g, split = setup
    result = run_experiment(g, split, ["logreg", "sage"], [0, 1], TrainConfig(epochs=20))
    assert set(result["summary"]) == {"logreg", "sage"}
    assert 0.0 <= result["summary"]["sage"]["pr_auc"]["mean"] <= 1.0
    leaky = type(split)(split.name, split.train, split.val, split.test | split.train)
    with pytest.raises(RuntimeError, match="leakage audit"):
        run_experiment(g, leaky, ["logreg"], [0])


def test_evaluate_uses_validation_threshold(setup):
    g, split = setup
    pred = run_model("logreg", g, split, seed=0)
    m = evaluate(pred, g, split)
    assert set(m) >= {"pr_auc", "roc_auc", "f1", "mcc", "precision_at_100", "ece"}
    assert "gat" in available_models()


def test_drift_study_rolling_retraining_recovers_after_concept_shift():
    from graphshield_lab.drift import drift_study

    g = make_synthetic_graph(num_steps=30, nodes_per_step=250, num_features=10,
                             labelled_fraction=0.9, shift_step=17, shift_strength=1.0, seed=11)
    res = drift_study(g, ["logreg"], [0], window=10, horizon=4, static_train_end=10,
                      static_val_end=12, cfg=TrainConfig(epochs=30))
    ok = [r for r in res["rows"] if "gain" in r]
    assert ok and all(0 <= r["static_pr_auc"] <= 1 and 0 <= r["rolling_pr_auc"] <= 1 for r in ok)
    # blocks fully after the shift: a model retrained on post-shift data should win
    late = [r for r in ok if r["block"][0] >= 25]
    assert late and np.mean([r["gain"] for r in late]) > 0.05
    assert res["summary"]["logreg"]["blocks"] == len(ok)
    # every block starts after the static validation window
    assert all(r["block"][0] > 12 for r in ok)


def test_drift_cli(tmp_path, capsys):
    from graphshield_data.store import save_graph

    from graphshield_lab.cli import main as lab_main

    g = make_synthetic_graph(num_steps=24, nodes_per_step=150, num_features=8,
                             labelled_fraction=0.9, shift_step=15, shift_strength=1.0, seed=3)
    save_graph(g, tmp_path / "d")
    code = lab_main(["drift", "--data", str(tmp_path / "d"), "--models", "logreg", "--seeds", "0",
                     "--window", "8", "--horizon", "3", "--static-train-end", "8",
                     "--static-val-end", "10", "--out", str(tmp_path / "o.json")])
    assert code == 0 and "rolling wins" in capsys.readouterr().out


def test_static_study_runs_end_to_end_with_paired_contrasts(tmp_path, capsys):
    from graphshield_data.store import save_graph

    from graphshield_lab.cli import main as lab_main
    from graphshield_lab.static import render_static, run_static_study

    g = make_synthetic_graph(num_steps=3, nodes_per_step=200, num_features=8,
                             labelled_fraction=0.9, seed=4)
    res = run_static_study(g, ["logreg", "hgb", "sage"], [0, 1, 2], TrainConfig(epochs=25),
                           references=("logreg",))
    assert set(res["summary"]) == {"logreg", "hgb", "sage"} and res["seeds"] == [0, 1, 2]
    c = res["pr_auc_contrasts"]["logreg"]
    assert set(c) == {"hgb", "sage"} and all(v["n"] == 3 for v in c.values())
    assert all(v["ci_low"] <= v["mean_diff"] <= v["ci_high"] for v in c.values())
    text = render_static(res)
    assert "| sage |" in text and "paired by seed" in text
    save_graph(g, tmp_path / "d")
    code = lab_main(["static", "--data", str(tmp_path / "d"), "--models", "logreg", "hgb",
                     "--seeds", "0", "1", "--epochs", "10", "--out", str(tmp_path / "o.json")])
    assert code == 0 and (tmp_path / "o.json").exists()
    assert "PR-AUC" in capsys.readouterr().out


def test_tuning_pool_never_overlaps_evaluation_labels():
    from graphshield_lab.tune import split_tuning_pool

    g = make_synthetic_graph(num_steps=3, nodes_per_step=300, num_features=6, labelled_fraction=0.9,
                             seed=8)
    tune_g, eval_g = split_tuning_pool(g, 0.25, seed=1)
    tuned, evald = tune_g.labels >= 0, eval_g.labels >= 0
    assert not (tuned & evald).any()  # disjoint label sets
    assert ((tuned | evald) == (g.labels >= 0)).all()  # together they cover every label
    assert abs(tuned.sum() / (g.labels >= 0).sum() - 0.25) < 0.02
    assert (tune_g.labels[tuned] == g.labels[tuned]).all()


def test_tune_model_returns_best_params_and_runs_in_study(tmp_path):
    from graphshield_lab.static import run_static_study
    from graphshield_lab.tune import apply_best, split_tuning_pool, tune_model

    g = make_synthetic_graph(num_steps=3, nodes_per_step=300, num_features=6, labelled_fraction=0.9,
                             seed=9)
    tg, eg = split_tuning_pool(g, 0.3)
    base = TrainConfig(epochs=15, patience=15)
    t_tab = tune_model(tg, "logreg", n_trials=3, base_cfg=base)
    t_nn = tune_model(tg, "sage", n_trials=2, base_cfg=base)
    assert set(t_tab["best"]) == {"C"} and len(t_tab["trials"]) == 3
    assert t_tab["best_score"] == max(t["val_pr_auc"] for t in t_tab["trials"])
    assert {"hidden", "lr", "dropout"} <= set(t_nn["best"])
    cfg_nn, params_nn = apply_best("sage", t_nn, base)
    cfg_tab, params_tab = apply_best("logreg", t_tab, base)
    assert params_nn is None and cfg_nn.hidden == t_nn["best"]["hidden"]
    assert params_tab == t_tab["best"]
    res = run_static_study(eg, ["logreg", "sage"], [0, 1], base, model_cfgs={"sage": cfg_nn},
                           model_params={"logreg": params_tab}, references=("logreg",))
    assert res["tuned"] and set(res["summary"]) == {"logreg", "sage"}
