"""Loading, saving and running a trained model bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from graphshield_lab.layers import to_undirected
from graphshield_lab.models import NodeClassifier

FORMAT_VERSION = 1


@dataclass
class ModelBundle:
    model: NodeClassifier
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float
    undirected: bool

    @property
    def num_features(self) -> int:
        return int(self.feature_mean.shape[0])

    def _prepare(self, features: np.ndarray, edge_index: np.ndarray):
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError(f"features must have shape (N, {self.num_features})")
        sd = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        x = torch.from_numpy(((features - self.feature_mean) / sd).astype(np.float32))
        ei = torch.as_tensor(edge_index, dtype=torch.long).reshape(2, -1)
        if ei.numel() and (ei.min() < 0 or ei.max() >= x.shape[0]):
            raise ValueError("edge_index refers to a node outside the supplied features")
        return x, (to_undirected(ei) if self.undirected else ei)

    @torch.no_grad()
    def score(self, features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Illicit probability for every node of the supplied (sub)graph."""
        x, ei = self._prepare(features, edge_index)
        self.model.eval()
        return torch.sigmoid(self.model(x, ei)).numpy()

    def saliency(self, features: np.ndarray, edge_index: np.ndarray, node: int):
        """Gradient-based attribution of ``node``'s logit to every node's input features.

        Returns ``(feature_importance, node_influence)``: the first is |d logit / d x| summed
        over all nodes per feature; the second is the same quantity summed over features per
        node, i.e. which transactions in the neighbourhood drive the score.
        """
        x, ei = self._prepare(features, edge_index)
        x.requires_grad_(True)
        self.model.eval()
        self.model(x, ei)[node].backward()
        grad = (x.grad * x.detach()).abs()  # gradient x input
        return grad.sum(0).numpy(), grad.sum(1).numpy()


def save_bundle(artifact: dict, path: str | Path) -> None:
    """Persist the ``Prediction.artifact`` produced by graphshield-model-lab."""
    if artifact is None:
        raise ValueError("this model type does not produce a servable artifact")
    torch.save(artifact, path)


def load_bundle(path: str | Path) -> ModelBundle:
    art = torch.load(path, map_location="cpu", weights_only=False)
    if art.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported bundle format {art.get('format_version')!r}")
    model = NodeClassifier(
        art["kind"], art["in_dim"], art["hidden"], art["num_layers"], art["dropout"], art["heads"]
    )
    model.load_state_dict(art["state_dict"])
    return ModelBundle(
        model=model,
        feature_mean=np.asarray(art["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(art["feature_std"], dtype=np.float32),
        threshold=float(art["threshold"]),
        undirected=bool(art["undirected"]),
    )
