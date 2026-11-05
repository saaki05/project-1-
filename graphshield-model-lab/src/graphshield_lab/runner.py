"""Training and evaluation protocol.

Protocol (identical for every model):

1. Training uses only the subgraph of nodes whose time step is <= the last training step.
2. Feature standardisation statistics come from training-split nodes only.
3. Early stopping and the decision threshold use the validation split only.
4. The test split is touched once, after those choices are fixed.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from graphshield_data import GraphData, Split
from torch import nn

from graphshield_lab import metrics as M
from graphshield_lab.baselines import BASELINES, make_baseline, neighbour_mean
from graphshield_lab.layers import to_undirected
from graphshield_lab.models import GNN_KINDS, NodeClassifier


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    heads: int = 4
    lr: float = 5e-3
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 30
    undirected: bool = True


@dataclass
class Prediction:
    model: str
    seed: int
    val_prob: np.ndarray
    test_prob: np.ndarray
    threshold: float
    artifact: dict | None = None  # serialisable model bundle, GNN/MLP models only


def available_models() -> list[str]:
    names = list(BASELINES) + [f"{b}+agg" for b in BASELINES] + ["mlp", *GNN_KINDS]
    return names


def _standardise(x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    mu = x[fit_mask].mean(axis=0)
    sd = x[fit_mask].std(axis=0)
    return ((x - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)


def _threshold_and_probs(y_val, val_prob):
    return M.best_f1_threshold(y_val, val_prob)


def run_model(
    name: str, graph: GraphData, split: Split, seed: int, cfg: TrainConfig | None = None,
    params: dict | None = None,
) -> Prediction:
    """Train ``name`` under the protocol and return validation/test probabilities."""
    cfg = cfg or TrainConfig()
    train_end = int(graph.timestep[split.train].max())
    test_end = int(graph.timestep[split.test].max())
    visible = graph.timestep <= test_end
    sub, kept = graph.subgraph(visible)
    pos = {int(g): i for i, g in enumerate(kept)}
    remap = np.vectorize(pos.get)
    tr = remap(np.flatnonzero(split.train))
    va = remap(np.flatnonzero(split.val))
    te = remap(np.flatnonzero(split.test))
    train_nodes = sub.timestep <= train_end
    y = sub.labels.astype(np.float32)

    x = _standardise(sub.features, np.isin(np.arange(sub.num_nodes), tr))
    ei = sub.edge_index

    if name in BASELINES or name.endswith("+agg"):
        base = name.removesuffix("+agg")
        feats = x
        if name.endswith("+agg"):
            # neighbourhood computed on the past-only graph for training, full visible graph
            # for evaluation; edges stay within time steps so both agree on train nodes.
            feats = np.concatenate([x, neighbour_mean(x, ei)], axis=1)
        clf = make_baseline(base, seed, params).fit(feats[tr], y[tr].astype(int))
        return _finish(name, seed, clf.predict_proba(feats[va])[:, 1],
                       clf.predict_proba(feats[te])[:, 1], y[va])

    torch.manual_seed(seed)
    np.random.seed(seed)
    xt = torch.from_numpy(x)
    eit = torch.from_numpy(ei.astype(np.int64))
    if cfg.undirected:
        eit = to_undirected(eit)
    # training graph: only nodes up to the last training step
    tsub, tkept = sub.subgraph(train_nodes)
    tei = torch.from_numpy(tsub.edge_index.astype(np.int64))
    if cfg.undirected:
        tei = to_undirected(tei)
    train_x = xt[torch.from_numpy(tkept)]
    tr_local = torch.from_numpy(np.searchsorted(tkept, tr))
    yt_train = torch.from_numpy(y[tkept])[tr_local]

    model = NodeClassifier(name, x.shape[1], cfg.hidden, cfg.num_layers, cfg.dropout, cfg.heads)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_w = float((1 - yt_train).sum() / yt_train.sum().clamp_min(1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))

    best_score, best_state, stale = -1.0, None, 0
    y_val = y[va]
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(train_x, tei)[tr_local], yt_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(xt, eit))[va].numpy()
        score = M.pr_auc(y_val, val_prob)
        if score > best_score:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(xt, eit)).numpy()
    pred = _finish(name, seed, prob[va], prob[te], y_val)
    pred.artifact = {
        "format_version": 1,
        "kind": name,
        "in_dim": int(x.shape[1]),
        "hidden": cfg.hidden,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "heads": cfg.heads,
        "undirected": cfg.undirected,
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "feature_mean": sub.features[tr].mean(axis=0),
        "feature_std": sub.features[tr].std(axis=0),
        "threshold": pred.threshold,
    }
    return pred


def _finish(name, seed, val_prob, test_prob, y_val) -> Prediction:
    return Prediction(name, seed, val_prob, test_prob, _threshold_and_probs(y_val, val_prob))


def evaluate(pred: Prediction, graph: GraphData, split: Split) -> dict[str, float]:
    """Test-set metrics using the validation-selected threshold."""
    y = graph.labels[split.test].astype(int)
    p = pred.test_prob
    return {
        "pr_auc": M.pr_auc(y, p),
        "roc_auc": M.roc_auc(y, p),
        "f1": M.f1_at(y, p, pred.threshold),
        "mcc": M.mcc_at(y, p, pred.threshold),
        "precision_at_100": M.precision_at_k(y, p, 100),
        "ece": M.ece(y, p),
    }


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
