"""Static checks that a split does not leak information between train and evaluation nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graphshield_data.graph import ILLICIT, UNKNOWN, GraphData
from graphshield_data.splits import Split

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity == ERROR for f in self.findings)

    def add(self, severity: str, code: str, message: str) -> None:
        self.findings.append(Finding(severity, code, message))

    def format(self) -> str:
        lines = [f"[{f.severity.upper():7}] {f.code}: {f.message}" for f in self.findings]
        lines.append("RESULT: " + ("OK" if self.ok else "FAILED"))
        return "\n".join(lines)


def _row_hashes(x: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in np.ascontiguousarray(x)}


def audit_split(g: GraphData, split: Split) -> AuditReport:
    """Check a split for overlap, temporal order, label validity, edge and duplicate leakage."""
    rep = AuditReport()
    masks = {"train": split.train, "val": split.val, "test": split.test}

    for name, m in masks.items():
        if m.shape != (g.num_nodes,):
            rep.add(ERROR, "mask-shape", f"{name} mask has wrong shape {m.shape}")
            return rep

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = int((masks[a] & masks[b]).sum())
        if n:
            rep.add(ERROR, "overlap", f"{n} nodes appear in both {a} and {b}")

    for name, m in masks.items():
        if not m.any():
            rep.add(ERROR, "empty", f"{name} split is empty")
            continue
        if (g.labels[m] == UNKNOWN).any():
            rep.add(ERROR, "unlabelled", f"{name} contains nodes with unknown labels")
        pos = int((g.labels[m] == ILLICIT).sum())
        rate = pos / int(m.sum())
        if pos == 0:
            rep.add(ERROR, "no-positives", f"{name} contains no illicit nodes")
        else:
            rep.add(INFO, "class-balance", f"{name}: n={int(m.sum())}, illicit rate={rate:.4f}")

    if split.check_time and split.train.any() and split.test.any():
        if g.timestep[split.train].max() >= g.timestep[split.test].min():
            rep.add(ERROR, "temporal-order", "training nodes are not strictly earlier than test")
        if split.val.any() and g.timestep[split.train].max() >= g.timestep[split.val].min():
            rep.add(ERROR, "temporal-order", "training nodes are not strictly earlier than val")

    src, dst = g.edge_index
    train_nodes = split.train
    eval_nodes = split.val | split.test
    cross = int(((train_nodes[src] & eval_nodes[dst]) | (eval_nodes[src] & train_nodes[dst])).sum())
    if cross:
        rep.add(
            WARNING,
            "cross-split-edges",
            f"{cross} edges connect train and evaluation nodes; train on "
            "GraphData.subgraph(train_mask) to avoid transductive leakage",
        )
    else:
        rep.add(INFO, "cross-split-edges", "no edges connect train and evaluation nodes")
    if not split.check_time:
        rep.add(INFO, "static-split", "no temporal ordering was checked (static graph); "
                "evaluation is transductive")

    if split.train.any() and split.test.any():
        dup = len(_row_hashes(g.features[split.train]) & _row_hashes(g.features[split.test]))
        if dup:
            rep.add(
                WARNING, "duplicate-features", f"{dup} identical feature rows in train and test"
            )
    return rep
