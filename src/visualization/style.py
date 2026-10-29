"""Shared matplotlib style so every figure in the report looks consistent."""
import matplotlib.pyplot as plt

PALETTE = {
    "treatment": "#2563eb",
    "control": "#94a3b8",
    "positive": "#16a34a",
    "negative": "#dc2626",
    "neutral": "#6b7280",
    "accent": "#f59e0b",
}

MODEL_COLORS = {
    "S-Learner": "#2563eb",
    "T-Learner": "#16a34a",
    "X-Learner": "#f59e0b",
    "DR-Learner": "#dc2626",
    "Causal Forest": "#7c3aed",
    "Random": "#94a3b8",
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
