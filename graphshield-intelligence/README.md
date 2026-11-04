# graphshield-intelligence

Serving layer for GraphShield: loads a trained `graphshield-model-lab` model and exposes
scoring, explanation and drift monitoring over HTTP.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | model kind, expected feature count, decision threshold |
| `POST /score` | illicit probability and alert flag for every node of a submitted subgraph |
| `POST /explain` | probability, top features and per-node influence for one node, using gradient × input saliency |
| `POST /drift` | per-feature PSI and KS statistics between a reference and a current feature batch |

Requests carry raw (unstandardised) features and an edge list `[[src, dst], ...]`; the
service applies the training-time standardisation stored in the model bundle. Malformed shapes
and out-of-range indices return HTTP 422.

## Interface contract

Input: a bundle written by `save_bundle(prediction.artifact, path)` from
`graphshield-model-lab` (`format_version` 1; loading refuses other versions).

## Install and run

```bash
git clone https://github.com/saaki05/project-1- && cd project-1-
pip install -e graphshield-data-engine -e graphshield-model-lab
cd graphshield-intelligence
pip install -e ".[dev]"
pytest
graphshield-serve --bundle path/to/model.pt --port 8000
```

Docker (build from the repository root so the sibling packages are available):

```bash
docker build -f graphshield-intelligence/Dockerfile -t graphshield .
docker run -p 8000:8000 -v "$PWD/models:/models" graphshield
```

## Notes and limitations

- Gradient × input saliency is a fast first-order attribution, not a faithful causal
  explanation. A perturbation-based edge-mask explainer is planned and not yet implemented.
- The bundle uses `torch.load(weights_only=False)`; only load bundles you produced yourself.
- PSI/KS thresholds (0.1 / 0.25) are conventional rules of thumb, not calibrated to any data
  set.
- The Dockerfile has not been built in the development environment.

## License

MIT.
