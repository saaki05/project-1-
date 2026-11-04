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
