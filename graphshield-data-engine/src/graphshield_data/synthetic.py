"""Synthetic temporal graphs with the same structure as the Elliptic data.

These graphs exist for tests, smoke runs and demonstrating drift handling. They are NOT a
substitute for real data: no conclusion about real fraud detection may be drawn from them.
"""

from __future__ import annotations

import numpy as np

from graphshield_data.graph import UNKNOWN, GraphData


def make_synthetic_graph(
    num_steps: int = 20,
    nodes_per_step: int = 200,
    num_features: int = 16,
    illicit_rate: float = 0.12,
    labelled_fraction: float = 0.5,
    homophily: float = 0.8,
    avg_degree: float = 2.0,
    signal: float = 1.0,
    shift_step: int | None = None,
    shift_strength: float = 1.5,
    seed: int = 0,
) -> GraphData:
    """Generate a temporal graph; edges stay within a time step.

    Illicit nodes have a shifted feature mean. If ``shift_step`` is given, from that step on
    the informative directions change, mimicking the drift that follows a market shutdown.
    """
    rng = np.random.default_rng(seed)
    n = num_steps * nodes_per_step
    n_info = max(2, num_features // 2)

    direction_a = rng.normal(size=n_info)
    direction_a /= np.linalg.norm(direction_a)
    direction_b = rng.normal(size=n_info)
    direction_b /= np.linalg.norm(direction_b)

    timestep = np.repeat(np.arange(1, num_steps + 1, dtype=np.int16), nodes_per_step)
    true_class = (rng.random(n) < illicit_rate).astype(np.int8)

    features = rng.normal(size=(n, num_features)).astype(np.float32)
    shifted = np.zeros(n, dtype=bool) if shift_step is None else timestep >= shift_step
    mean_dir = np.where(shifted[:, None], direction_b[None, :], direction_a[None, :])
    scale = np.where(shifted, signal / shift_strength, signal)
    features[:, :n_info] += (true_class[:, None] * scale[:, None] * mean_dir * 3.0).astype(
        np.float32
    )

    src_list, dst_list = [], []
    for t in range(num_steps):
        lo = t * nodes_per_step
        idx = np.arange(lo, lo + nodes_per_step)
        cls = true_class[idx]
        pools = {0: idx[cls == 0], 1: idx[cls == 1]}
        n_edges = int(avg_degree * nodes_per_step)
        src = rng.choice(idx, size=n_edges)
        same = rng.random(n_edges) < homophily
        dst = np.empty(n_edges, dtype=np.int64)
        for k in range(n_edges):
            c = true_class[src[k]] if same[k] else 1 - true_class[src[k]]
            pool = pools[int(c)] if pools[int(c)].size else idx
            dst[k] = rng.choice(pool)
        keep = src != dst
        src_list.append(src[keep])
        dst_list.append(dst[keep])

    labels = true_class.copy()
    labels[rng.random(n) > labelled_fraction] = UNKNOWN
    g = GraphData(
        node_ids=np.arange(1_000_000, 1_000_000 + n, dtype=np.int64),
        timestep=timestep,
        features=features,
        labels=labels.astype(np.int8),
        edge_index=np.stack([np.concatenate(src_list), np.concatenate(dst_list)]).astype(np.int64),
    )
    g.validate()
    return g
