"""Temporal train/validation/test splits over labelled nodes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from graphshield_data.graph import UNKNOWN, GraphData


@dataclass(frozen=True)
class Split:
    """Boolean node masks. Masks only ever select nodes with a known label."""

    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    check_time: bool = True  # False for static (non-temporal) graphs


def _range_mask(g: GraphData, steps: tuple[int, int]) -> np.ndarray:
    lo, hi = steps
    if lo > hi:
        raise ValueError(f"empty step range {steps}")
    return (g.timestep >= lo) & (g.timestep <= hi) & (g.labels != UNKNOWN)


def temporal_split(
    g: GraphData,
    train: tuple[int, int],
    val: tuple[int, int],
    test: tuple[int, int],
    name: str = "temporal",
) -> Split:
    """Split by inclusive time-step ranges, requiring train < val < test strictly in time."""
    if not (train[1] < val[0] and val[1] < test[0]):
        raise ValueError("ranges must satisfy train < val < test without overlap")
    return Split(
        name=name,
        train=_range_mask(g, train),
        val=_range_mask(g, val),
        test=_range_mask(g, test),
    )


def rolling_windows(
    g: GraphData,
    window: int,
    horizon: int,
    val_steps: int = 2,
    stride: int = 1,
) -> Iterator[Split]:
    """Yield sliding-window splits for drift-aware retraining.

    For every start step ``s``: train covers ``[s, s+window-val_steps)``, validation the
    following ``val_steps`` steps, and test the next ``horizon`` steps.
    """
    if window <= val_steps or horizon < 1 or stride < 1:
        raise ValueError("need window > val_steps, horizon >= 1 and stride >= 1")
    first, last = int(g.timestep.min()), int(g.timestep.max())
    start = first
    while start + window + horizon - 1 <= last:
        tr_end = start + window - val_steps - 1
        va_end = start + window - 1
        yield temporal_split(
            g,
            train=(start, tr_end),
            val=(tr_end + 1, va_end),
            test=(va_end + 1, va_end + horizon),
            name=f"window@{start}",
        )
        start += stride


def stratified_split(g: GraphData, fractions=(0.4, 0.2, 0.4), seed: int = 0,
                     name: str = "stratified") -> Split:
    """Random split of *labelled* nodes that preserves the class ratio in every part.

    For graphs without time information (``check_time=False`` is recorded on the split so the
    auditor does not demand temporal ordering). Evaluation is transductive: the full graph
    structure is visible while only training labels are used for fitting.
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    masks = [np.zeros(g.num_nodes, dtype=bool) for _ in range(3)]
    for cls in (0, 1):
        idx = np.flatnonzero(g.labels == cls)
        if len(idx) < 3:
            raise ValueError(f"class {cls} has fewer than 3 labelled nodes")
        idx = rng.permutation(idx)
        n_tr = max(1, round(fractions[0] * len(idx)))
        n_va = max(1, round(fractions[1] * len(idx)))
        masks[0][idx[:n_tr]] = True
        masks[1][idx[n_tr:n_tr + n_va]] = True
        masks[2][idx[n_tr + n_va:]] = True
    return Split(name, masks[0], masks[1], masks[2], check_time=False)
