"""Generate every GraphShield figure from the result JSON files (no hard-coded numbers).

Usage: python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "graphshield-model-lab" / "results" / "amazon"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
fs.setup()

LABEL = {"logreg": "Logistic regression", "rf": "Random forest", "hgb": "Gradient boosting",
         "logreg+agg": "Logistic reg. + neighbours", "rf+agg": "Random forest + neighbours",
         "hgb+agg": "Gradient boosting + neighbours", "mlp": "MLP", "gcn": "GCN",
         "sage": "GraphSAGE", "gat": "GAT"}


def kind(name):
    if name in ("gcn", "sage", "gat"):
        return fs.ORANGE
    return fs.BLUE if name.endswith("+agg") else fs.GRAY


def model_comparison():
    d = json.load(open(RES / "static_upu_tuned.json"))
    order = sorted(d["summary"], key=lambda m: d["summary"][m]["pr_auc"]["mean"])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, m in enumerate(order):
        vals = [r["pr_auc"] for r in d["per_seed"][m]]
        mean = d["summary"][m]["pr_auc"]["mean"]
        ax.scatter(vals, [i] * len(vals), s=14, color=kind(m), alpha=0.35, zorder=2)
        ax.plot([mean - d["summary"][m]["pr_auc"]["std"], mean + d["summary"][m]["pr_auc"]["std"]],
                [i, i], color=kind(m), lw=2.4, zorder=3)
        ax.scatter([mean], [i], s=46, color=kind(m), zorder=4, edgecolor=fs.SURFACE, lw=1.2)
        ax.text(max(vals) + 0.012, i, f"{mean:.3f}", va="center", fontsize=8.5, color=fs.INK2)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    ax.set_xlabel("PR-AUC on the test split (10 random splits; dot = one split, bar = ±1 std)")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Amazon fraud graph: PR-AUC of every model after hyper-parameter search")
    ax.grid(axis="y", visible=False)
    for c, t in ((fs.BLUE, "tabular + neighbour mean"), (fs.GRAY, "tabular / MLP"),
                 (fs.ORANGE, "graph neural network")):
        ax.scatter([], [], color=c, label=t, s=30)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fs.save(fig, OUT / "model_comparison.png")


def contrasts():
    d = json.load(open(RES / "static_upu_tuned.json"))["pr_auc_contrasts"]["hgb"]
    names = sorted(d, key=lambda m: d[m]["mean_diff"])
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.axvline(0, color=fs.MUTED, lw=1)
    for i, m in enumerate(names):
        c = d[m]
        ax.plot([c["ci_low"], c["ci_high"]], [i, i], color=kind(m), lw=2.6)
        ax.scatter([c["mean_diff"]], [i], color=kind(m), s=42, zorder=3, edgecolor=fs.SURFACE)
        ax.text(1.02, i, f"{c['mean_diff']:+.3f}   {c['wins']}/{c['n']} splits better", va="center",
                ha="left", fontsize=8.3, color=fs.INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(names)), [LABEL[m] for m in names])
    ax.set_xlabel("PR-AUC difference to tuned gradient boosting (paired by split, 95% bootstrap CI)")
    ax.set_xlim(-0.42, 0.12)
    ax.set_title("PR-AUC difference to tuned gradient boosting, paired by split")
    ax.grid(axis="y", visible=False)
    fs.save(fig, OUT / "paired_contrasts.png")


def tuned_comparison():
    f = RES / "static_upu_tuned.json"
    if not f.exists():
        return
    t = json.load(open(f))
    d = json.load(open(RES / "static_upu.json"))
    order = sorted(t["summary"], key=lambda m: t["summary"][m]["pr_auc"]["mean"])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, m in enumerate(order):
        a, b = d["summary"][m]["pr_auc"]["mean"], t["summary"][m]["pr_auc"]["mean"]
        ax.plot([a, b], [i, i], color=fs.GRID, lw=3, zorder=1)
        ax.scatter([a], [i], facecolor="none", edgecolor=fs.INK2, linewidth=1.6, s=62, zorder=3,
                   label="default settings (hollow)" if i == 0 else None)
        ax.scatter([b], [i], color=kind(m), s=60, zorder=4, edgecolor=fs.SURFACE)
        ax.text(max(a, b) + 0.012, i, f"{b:.3f}  ({b - a:+.3f})", va="center", fontsize=8.5,
                color=fs.INK2)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("PR-AUC (mean of 10 splits). Hollow = default settings, filled = after random search.\nThe tuned run reserves 20% of the labelled nodes for tuning, so the two runs use different test nodes.")
    ax.set_title("Default settings against settings found by random search")
    ax.grid(axis="y", visible=False)
    for c, lab in ((fs.BLUE, "tabular + neighbour mean"), (fs.GRAY, "tabular / MLP"),
                   (fs.ORANGE, "graph neural network")):
        ax.scatter([], [], color=c, label=lab, s=30)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fs.save(fig, OUT / "tuned_comparison.png")


def neighbourhood():
    h = json.load(open(RES / "homophily.json"))
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    rels = [("upu", "UPU: same product"), ("uvu", "UVU: similar text")]
    base = h["upu"]["base_fraud_rate"]
    for i, (r, lab) in enumerate(rels):
        share = h[r]["fraud_neighbour_share_of_fraud_nodes"]
        ax.barh(i, share, color=fs.ORANGE, height=0.5)
        ax.text(share + 0.004, i, f"{share:.1%}", va="center", color=fs.INK2)
    ax.axvline(base, color=fs.INK2, ls="--", lw=1.2)
    ax.text(base + 0.004, 0.5, f"overall fraud rate {base:.1%}", fontsize=8.5, color=fs.INK2,
            va="center")
    ax.set_yticks(range(2), [lab for _, lab in rels])
    ax.set_xlim(0, 0.28)
    ax.set_xlabel("share of a fraud user's neighbours that are fraud")
    ax.set_title("Fraud users mostly connect to non-fraud users")
    ax.grid(axis="y", visible=False)
    fs.save(fig, OUT / "neighbourhood.png")


def architecture():
    fs.pipeline(OUT / "architecture.png", [
        ("data-engine", ["load Elliptic / Amazon", "validated graph", "temporal + stratified splits",
                         "leakage auditor", "Parquet + manifest"]),
        ("model-lab", ["GCN, SAGE, GAT (scratch)", "tabular baselines", "+ neighbour-mean features",
                       "paired seeds, bootstrap", "static + drift studies"]),
        ("intelligence", ["model bundle (v1)", "/score  /explain", "/drift  (PSI, KS)", "FastAPI service"]),
    ], "GraphShield: three packages, one-way dependencies", accent=fs.BLUE, width=10.5)


if __name__ == "__main__":
    model_comparison()
    contrasts()
    tuned_comparison()
    neighbourhood()
    architecture()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
