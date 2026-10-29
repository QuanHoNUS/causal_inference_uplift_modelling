"""Treatment assignment analysis: covariate balance (SMD), a treatment
prediction model, and propensity score diagnostics.

These answer the assignment's Section 6.1 questions directly:
  - Does treatment appear randomized?
  - Can customer features predict treatment assignment?
  - What evidence supports the conclusion?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src.utils.config import FEATURE_COLS, RANDOM_SEED, TREATMENT_COL


def standardized_mean_difference(df: pd.DataFrame, feature_cols=FEATURE_COLS, treatment_col=TREATMENT_COL) -> pd.DataFrame:
    """SMD = (mean_treated - mean_control) / pooled_std.

    |SMD| < 0.1 is the conventional "well balanced" threshold used in the
    causal inference literature (Austin, 2011).
    """
    rows = []
    t = df[df[treatment_col] == 1]
    c = df[df[treatment_col] == 0]
    for f in feature_cols:
        mt, mc = t[f].mean(), c[f].mean()
        vt, vc = t[f].var(ddof=1), c[f].var(ddof=1)
        pooled_std = np.sqrt((vt + vc) / 2)
        smd = (mt - mc) / pooled_std if pooled_std > 0 else np.nan
        rows.append({
            "feature": f,
            "mean_treated": mt,
            "mean_control": mc,
            "std_treated": np.sqrt(vt),
            "std_control": np.sqrt(vc),
            "smd": smd,
            "abs_smd": abs(smd),
            "balanced_lt_0.1": abs(smd) < 0.1,
        })
    out = pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)
    return out


def plot_smd_love_plot(smd_table: pd.DataFrame, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    order = smd_table.sort_values("abs_smd")
    colors = ["#dc2626" if not b else "#16a34a" for b in order["balanced_lt_0.1"]]
    ax.hlines(y=order["feature"], xmin=0, xmax=order["smd"], color="#94a3b8", linewidth=1)
    ax.scatter(order["smd"], order["feature"], color=colors, s=60, zorder=3)
    ax.axvline(0.1, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(-0.1, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized Mean Difference (Treated - Control)")
    ax.set_title("Covariate Balance: Love Plot")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def fit_propensity_model(
    df: pd.DataFrame,
    feature_cols=FEATURE_COLS,
    treatment_col=TREATMENT_COL,
    cv_folds: int = 3,
    seed: int = RANDOM_SEED,
    max_rows_for_cv: int | None = None,
) -> dict:
    """Fit a logistic-regression propensity model and evaluate, via
    cross-validated out-of-fold predictions, whether observable features
    predict treatment assignment.

    If treatment is truly randomized, AUC should be ~0.5 (chance) and the
    predicted propensity should cluster tightly around the true assignment
    probability, regardless of X.
    """
    work_df = df
    if max_rows_for_cv and len(df) > max_rows_for_cv:
        work_df = df.sample(max_rows_for_cv, random_state=seed)

    X = work_df[feature_cols].to_numpy()
    y = work_df[treatment_col].to_numpy()

    model = LogisticRegression(max_iter=200, n_jobs=-1)
    oof_pred = cross_val_predict(model, X, y, cv=cv_folds, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, oof_pred)

    # Fit final model on all (sampled) data for feature-importance inspection.
    model.fit(X, y)
    coef_table = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_[0],
        "abs_coefficient": np.abs(model.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "auc": auc,
        "propensity_scores": oof_pred,
        "true_treatment": y,
        "coef_table": coef_table,
        "empirical_treatment_rate": float(y.mean()),
        "n_used": len(work_df),
    }


def plot_propensity_diagnostics(prop_result: dict, save_path=None):
    ps = prop_result["propensity_scores"]
    y = prop_result["true_treatment"]
    emp_rate = prop_result["empirical_treatment_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(ps[y == 1], bins=60, alpha=0.55, density=True, color="#2563eb", label="Treated")
    axes[0].hist(ps[y == 0], bins=60, alpha=0.55, density=True, color="#94a3b8", label="Control")
    axes[0].axvline(emp_rate, color="black", linestyle="--", linewidth=1, label=f"Empirical rate={emp_rate:.3f}")
    axes[0].set_xlabel("Estimated propensity score  P(treatment=1 | X)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Propensity Score Distribution (AUC={prop_result['auc']:.4f})")
    axes[0].legend(fontsize=9)

    axes[1].hist(ps, bins=60, color="#6b7280")
    axes[1].axvline(emp_rate, color="black", linestyle="--", linewidth=1)
    span = ps.max() - ps.min()
    axes[1].set_title(f"All Propensity Scores (range={span:.4f})")
    axes[1].set_xlabel("Estimated propensity score")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def treatment_assignment_verdict(auc: float, smd_table: pd.DataFrame) -> dict:
    n_imbalanced = int((~smd_table["balanced_lt_0.1"]).sum())
    max_abs_smd = float(smd_table["abs_smd"].max())
    looks_randomized = auc < 0.55 and n_imbalanced == 0
    return {
        "auc": auc,
        "n_features_imbalanced_smd_gt_0.1": n_imbalanced,
        "max_abs_smd": max_abs_smd,
        "looks_randomized": looks_randomized,
    }
