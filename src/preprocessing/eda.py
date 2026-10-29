"""Exploratory data analysis utilities: schema, missingness, treatment ratio,
outcome rate, feature distributions, and feature correlations."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import CONVERSION_COL, FEATURE_COLS, TREATMENT_COL, VISIT_COL


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(("n_rows", len(df)))
    rows.append(("n_features", len(FEATURE_COLS)))
    rows.append(("treatment_ratio", df[TREATMENT_COL].mean()))
    rows.append(("control_ratio", 1 - df[TREATMENT_COL].mean()))
    rows.append(("overall_conversion_rate", df[CONVERSION_COL].mean()))
    rows.append(("overall_visit_rate", df[VISIT_COL].mean()))
    rows.append(("conversion_rate_treatment", df.loc[df[TREATMENT_COL] == 1, CONVERSION_COL].mean()))
    rows.append(("conversion_rate_control", df.loc[df[TREATMENT_COL] == 0, CONVERSION_COL].mean()))
    rows.append(("visit_rate_treatment", df.loc[df[TREATMENT_COL] == 1, VISIT_COL].mean()))
    rows.append(("visit_rate_control", df.loc[df[TREATMENT_COL] == 0, VISIT_COL].mean()))
    rows.append(("n_missing_total", int(df.isna().sum().sum())))
    rows.append(("n_duplicate_rows", int(df.duplicated().sum())))
    return pd.DataFrame(rows, columns=["metric", "value"])


def naive_lift_summary(df: pd.DataFrame, outcome_col: str = CONVERSION_COL) -> dict:
    """Naive (unadjusted) treatment-vs-control gap -- the number a marketer
    would compute before any causal reasoning. Used later to contrast
    against adjusted ATE estimates."""
    t_rate = df.loc[df[TREATMENT_COL] == 1, outcome_col].mean()
    c_rate = df.loc[df[TREATMENT_COL] == 0, outcome_col].mean()
    n_t = int((df[TREATMENT_COL] == 1).sum())
    n_c = int((df[TREATMENT_COL] == 0).sum())
    return {
        "outcome": outcome_col,
        "treated_rate": t_rate,
        "control_rate": c_rate,
        "absolute_lift": t_rate - c_rate,
        "relative_lift_pct": 100 * (t_rate - c_rate) / c_rate if c_rate > 0 else np.nan,
        "n_treated": n_t,
        "n_control": n_c,
    }


def feature_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS].describe().T


def plot_treatment_and_outcome_distribution(df: pd.DataFrame, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    df[TREATMENT_COL].value_counts(normalize=True).sort_index().plot(
        kind="bar", ax=axes[0], color=["#94a3b8", "#2563eb"]
    )
    axes[0].set_title("Treatment Assignment Ratio")
    axes[0].set_xticklabels(["Control (0)", "Treatment (1)"], rotation=0)
    axes[0].set_ylabel("Share of customers")

    rates = df.groupby(TREATMENT_COL)[CONVERSION_COL].mean()
    rates.plot(kind="bar", ax=axes[1], color=["#94a3b8", "#2563eb"])
    axes[1].set_title("Conversion Rate by Arm")
    axes[1].set_xticklabels(["Control (0)", "Treatment (1)"], rotation=0)
    axes[1].set_ylabel("P(conversion)")

    rates_v = df.groupby(TREATMENT_COL)[VISIT_COL].mean()
    rates_v.plot(kind="bar", ax=axes[2], color=["#94a3b8", "#2563eb"])
    axes[2].set_title("Visit Rate by Arm")
    axes[2].set_xticklabels(["Control (0)", "Treatment (1)"], rotation=0)
    axes[2].set_ylabel("P(visit)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_feature_distributions(df: pd.DataFrame, save_path=None, sample_n: int = 200_000, seed: int = 42):
    plot_df = df.sample(min(sample_n, len(df)), random_state=seed) if len(df) > sample_n else df
    n_feats = len(FEATURE_COLS)
    ncols = 4
    nrows = int(np.ceil(n_feats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, f in enumerate(FEATURE_COLS):
        ax = axes[i]
        for t, color, label in [(0, "#94a3b8", "Control"), (1, "#2563eb", "Treatment")]:
            vals = plot_df.loc[plot_df[TREATMENT_COL] == t, f]
            ax.hist(vals, bins=40, alpha=0.5, density=True, color=color, label=label)
        ax.set_title(f)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n_feats, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Distributions by Treatment Arm (sampled)", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_correlation(df: pd.DataFrame, save_path=None):
    corr = df[FEATURE_COLS + [TREATMENT_COL, CONVERSION_COL]].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature / Treatment / Outcome Correlation Matrix")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig, corr
