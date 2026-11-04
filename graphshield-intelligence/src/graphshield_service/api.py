"""FastAPI application exposing scoring, explanation and drift endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphshield_service.bundle import ModelBundle, load_bundle
from graphshield_service.drift import drift_report


class SubgraphRequest(BaseModel):
    features: list[list[float]] = Field(description="N x F raw (unstandardised) features")
    edges: list[list[int]] = Field(default_factory=list, description="[[src, dst], ...]")


class ExplainRequest(SubgraphRequest):
    node: int


class DriftRequest(BaseModel):
    reference: list[list[float]]
    current: list[list[float]]


def create_app(bundle_path: str | Path | None = None, bundle: ModelBundle | None = None) -> FastAPI:
    if bundle is None:
        if bundle_path is None:
            raise ValueError("provide bundle_path or bundle")
        bundle = load_bundle(bundle_path)
    app = FastAPI(title="GraphShield Intelligence", version="0.1.0")

    def parse(req: SubgraphRequest) -> tuple[np.ndarray, np.ndarray]:
        try:
            x = np.asarray(req.features, dtype=np.float32)
            e = np.asarray(req.edges, dtype=np.int64).reshape(-1, 2).T
            bundle._prepare(x, e)  # validate shapes and indices
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return x, e

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "num_features": bundle.num_features,
                "model": bundle.model.kind, "threshold": bundle.threshold}

    @app.post("/score")
    def score(req: SubgraphRequest) -> dict:
        x, e = parse(req)
        p = bundle.score(x, e)
        return {"probabilities": p.tolist(), "flagged": (p >= bundle.threshold).tolist(),
                "threshold": bundle.threshold}

    @app.post("/explain")
    def explain(req: ExplainRequest) -> dict:
        x, e = parse(req)
        if not 0 <= req.node < x.shape[0]:
            raise HTTPException(status_code=422, detail="node index out of range")
        feat, node = bundle.saliency(x, e, req.node)
        top = np.argsort(-feat)[:10]
        return {
            "probability": float(bundle.score(x, e)[req.node]),
            "method": "gradient x input saliency",
            "top_features": [{"feature": int(j), "importance": float(feat[j])} for j in top],
            "node_influence": node.tolist(),
        }

    @app.post("/drift")
    def drift(req: DriftRequest) -> dict:
        try:
            return drift_report(np.asarray(req.reference), np.asarray(req.current))
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

    return app
