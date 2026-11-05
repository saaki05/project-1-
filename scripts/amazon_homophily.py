"""Measure how fraud users connect in the Amazon relations; writes homophily.json."""
import json
import sys

from graphshield_data.amazon import load_amazon

mat, out = sys.argv[1], sys.argv[2]
res = {}
for rel in ("upu", "uvu"):
    g = load_amazon(mat, rel)
    s, d = g.edge_index
    lab = g.labels
    both = (lab[s] >= 0) & (lab[d] >= 0)
    base = float((lab[lab >= 0] == 1).mean())
    from_fraud = both & (lab[s] == 1)
    res[rel] = {
        "edges_among_labelled": int(both.sum()),
        "edge_homophily": float((lab[s][both] == lab[d][both]).mean()),
        "expected_if_random": base**2 + (1 - base) ** 2,
        "base_fraud_rate": base,
        "fraud_neighbour_share_of_fraud_nodes": float((lab[d][from_fraud] == 1).mean()),
    }
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=2))
