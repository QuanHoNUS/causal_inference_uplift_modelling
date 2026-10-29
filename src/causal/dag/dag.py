"""Causal DAG for the marketing-campaign business scenario.

Rendered with networkx + matplotlib (no external Graphviz binary required,
so the figure reproduces on any machine with just the Python stack).
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

NODE_ROLE_COLORS = {
    "treatment": "#2563eb",
    "outcome": "#dc2626",
    "confounder": "#16a34a",
    "mediator": "#f59e0b",
    "collider": "#7c3aed",
    "unobserved": "#94a3b8",
}

NODES = {
    "U": ("Unobserved intent /\nbudget / engagement", "unobserved"),
    "X": ("Customer features\n(f0-f11)", "confounder"),
    "T": ("Treatment\n(campaign assignment)", "treatment"),
    "E": ("Exposure\n(ad actually delivered)", "mediator"),
    "V": ("Visit\n(site visit)", "collider"),
    "Y": ("Conversion\n(purchase)", "outcome"),
}

EDGES = [
    ("U", "X", "solid"),     # unobserved traits shape observed features
    ("U", "T", "dashed"),    # ONLY present under an observational assignment mechanism
    ("U", "Y", "solid"),     # unobserved traits affect the outcome directly
    ("X", "T", "dashed"),    # ONLY present under an observational assignment mechanism
    ("X", "Y", "solid"),     # features affect outcome directly (e.g. price sensitivity)
    ("T", "E", "solid"),
    ("E", "V", "solid"),
    ("T", "V", "solid"),     # campaign can drive visits even without confirmed exposure logging
    ("V", "Y", "solid"),
    ("U", "V", "solid"),     # unobserved browsing propensity also drives visits -> V is a collider on the T->Y path
]

DISCUSSION = {
    "confounders": (
        "X (the 12 anonymized customer features) is the primary *observed* "
        "confounder candidate: if it affects both who gets treated (in an "
        "observational assignment mechanism) and the outcome, it opens a "
        "back-door path T <- X -> Y that must be closed by adjustment "
        "(regression adjustment / IPW / DR). U (unobserved intent, budget, "
        "engagement) is a potential *unobserved* confounder -- if it "
        "affects both assignment and outcome, no amount of adjustment on X "
        "alone identifies the effect; this is the key threat to validity "
        "in the observational scenario discussed in Section 6.3."
    ),
    "mediators": (
        "E (exposure -- whether the ad was actually delivered/seen) and V "
        "(site visit) both sit on the causal path FROM the campaign TO "
        "conversion: T -> E -> V -> Y. They explain *how* the campaign "
        "works, but must never be conditioned on when estimating the "
        "causal effect of T on Y -- adjusting for a mediator blocks part "
        "of the very effect being measured and biases the estimate toward "
        "zero (this is why the ATE analysis in this project always "
        "compares outcomes by the ORIGINAL treatment assignment T, i.e. "
        "an intent-to-treat analysis, never by realized exposure or visit)."
    ),
    "colliders": (
        "V (Visit) is simultaneously a mediator (T -> V) and a collider on "
        "the path T -> V <- U -> Y, because unobserved browsing propensity "
        "U also drives visits. A common real-world mistake is to compute "
        "'conversion rate among visitors' by arm -- this conditions on the "
        "collider V, opening a spurious U-induced association between T "
        "and Y that does not reflect the campaign's true causal effect. "
        "This project deliberately avoids conditioning on V or E anywhere "
        "in the ATE/CATE estimation."
    ),
    "identifiability": (
        "Scenario 1 (randomized, the ACTUAL Criteo assignment mechanism): "
        "the dashed U->T and X->T edges are absent by design, so the "
        "back-door path through X and U is closed automatically and "
        "ATE = E[Y|T=1] - E[Y|T=0] is identified without any adjustment. "
        "Scenario 2 (hypothetical observational assignment, e.g. a "
        "marketer manually targets high-value segments): if only X drives "
        "targeting, the effect remains identifiable via back-door "
        "adjustment on X (regression adjustment / IPW / DR, conditional on "
        "positivity and no unmeasured confounding). If U also drives "
        "targeting AND affects Y outside of X, the effect is not "
        "identifiable from observational data alone -- adjustment on X is "
        "necessary but not sufficient, and the residual bias direction/"
        "magnitude is unknown without an instrument or a natural "
        "experiment."
    ),
}


def build_dag() -> nx.DiGraph:
    g = nx.DiGraph()
    for node, (label, role) in NODES.items():
        g.add_node(node, label=label, role=role)
    for src, dst, style in EDGES:
        g.add_edge(src, dst, style=style)
    return g


def plot_dag(save_path=None):
    g = build_dag()
    pos = {
        "U": (0.0, 2.2),
        "X": (0.0, 0.0),
        "T": (1.6, 0.0),
        "E": (3.2, 0.0),
        "V": (4.8, 0.0),
        "Y": (6.4, 0.0),
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for node, (label, role) in NODES.items():
        color = NODE_ROLE_COLORS[role]
        x, y = pos[node]
        ax.scatter([x], [y], s=2600, color=color, zorder=3, edgecolors="white", linewidths=2)
        ax.text(x, y, node, ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.45, label, ha="center", va="top", fontsize=8.5, zorder=4)

    for src, dst, style in EDGES:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        linestyle = "dashed" if style == "dashed" else "solid"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color="#374151", lw=1.6,
                linestyle=linestyle, shrinkA=22, shrinkB=22,
                connectionstyle="arc3,rad=0.12" if (src, dst) in [("T", "V")] else "arc3,rad=0.0",
            ),
            zorder=2,
        )

    legend_handles = [mpatches.Patch(color=c, label=role.capitalize()) for role, c in NODE_ROLE_COLORS.items()]
    legend_handles.append(plt.Line2D([0], [0], color="#374151", lw=1.6, linestyle="dashed", label="Observational-scenario only"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=1)

    ax.set_xlim(-1.4, 7.6)
    ax.set_ylim(-1.4, 3.0)
    ax.axis("off")
    ax.set_title(
        "Causal DAG: Marketing Campaign -> Conversion\n"
        "(dashed edges exist only under an observational assignment mechanism, Scenario 2)",
        fontsize=12,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
