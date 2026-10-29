"""Construct a synthetic "observational" version of the (actually
randomized) Criteo data, to make Section 6.3's Scenario 2 concrete and to
stress-test the estimators in the ablation study (Section 7.4).

Method: "confounding by biased sub-sampling" (standard technique for
turning an RCT into a semi-synthetic observational benchmark, e.g. used in
Kallus & Zhou 2018, Athey & Imbens tutorials). We keep every row's
observed (X, T, Y) untouched -- only the *retention probability* depends
on a chosen covariate and the row's own arm:

    P(keep | X, T=1) = sigmoid(+beta * clipped_standardize(X_confounder))
    P(keep | X, T=0) = sigmoid(-beta * clipped_standardize(X_confounder))

so the surviving treated group skews toward high values of the confounder
and the surviving control group skews toward low values. Since the
confounder (f4) is genuinely correlated with conversion (Pearson r=+0.30,
the second-strongest of all 12 features -- see EDA), the resulting
sub-sample has a real, observed, X-driven confounding structure: X now
affects both "who got treated" (via biased retention) and Y directly.

Biased retention also changes the X distribution, so the full-population
RCT ATE is not the correct benchmark. We therefore compute a weighted RCT
reference for the exact target population induced by the marginal
retention probability. This separates confounding bias from estimand drift.
Clipping the standardized confounder preserves practical overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import RANDOM_SEED, TREATMENT_COL

CONFOUNDER_FEATURE = "f4"  # r=+0.30 with conversion -- see EDA correlation ranking
DEFAULT_BETA = 1.0
DEFAULT_Z_CLIP = 2.0
Z_95 = 1.959963984540054


def _selection_probabilities(
    df: pd.DataFrame,
    confounder: str,
    beta: float,
    z_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[confounder].to_numpy(dtype=float)
    z = np.clip((x - x.mean()) / x.std(), -z_clip, z_clip)
    p_keep_treated = 1 / (1 + np.exp(-beta * z))
    p_keep_control = 1 / (1 + np.exp(beta * z))
    return z, p_keep_treated, p_keep_control


def make_confounded_observational_sample(
    df: pd.DataFrame,
    confounder: str = CONFOUNDER_FEATURE,
    beta: float = DEFAULT_BETA,
    z_clip: float = DEFAULT_Z_CLIP,
    max_rows: int | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    _, p_keep_treated, p_keep_control = _selection_probabilities(df, confounder, beta, z_clip)
    t = df[TREATMENT_COL].to_numpy()

    p_keep = np.where(t == 1, p_keep_treated, p_keep_control)
    keep = rng.random(len(df)) < p_keep

    out = df.loc[keep].reset_index(drop=True)
    if max_rows and len(out) > max_rows:
        out = out.sample(max_rows, random_state=seed).reset_index(drop=True)
    return out


def target_population_rct_reference(
    df: pd.DataFrame,
    outcome_col: str,
    confounder: str = CONFOUNDER_FEATURE,
    beta: float = DEFAULT_BETA,
    z_clip: float = DEFAULT_Z_CLIP,
) -> dict:
    """Estimate the target-population ATE from the original randomized data.

    The target population has density proportional to the marginal
    retention probability q(X) = e*s1(X) + (1-e)*s0(X). A weighted
    difference in randomized-arm means therefore estimates the same ATE
    targeted by adjustment estimators in the retained observational sample.
    """
    _, p_keep_treated, p_keep_control = _selection_probabilities(df, confounder, beta, z_clip)
    t = df[TREATMENT_COL].to_numpy()
    y = df[outcome_col].to_numpy(dtype=float)
    e = float(t.mean())
    target_weight = e * p_keep_treated + (1 - e) * p_keep_control

    treated_weight = target_weight * t
    control_weight = target_weight * (1 - t)
    mu1 = float(np.sum(treated_weight * y) / np.sum(treated_weight))
    mu0 = float(np.sum(control_weight * y) / np.sum(control_weight))
    ate = mu1 - mu0

    influence_treated = treated_weight * (y - mu1) / treated_weight.mean()
    influence_control = control_weight * (y - mu0) / control_weight.mean()
    influence = influence_treated - influence_control
    se = float(influence.std(ddof=1) / np.sqrt(len(df)))
    return {
        "estimand": "ATE in the marginal-retention target population",
        "ate": ate,
        "se": se,
        "ci_low": ate - Z_95 * se,
        "ci_high": ate + Z_95 * se,
        "treated_rate": mu1,
        "control_rate": mu0,
        "mean_target_weight": float(target_weight.mean()),
        "beta": beta,
        "z_clip": z_clip,
    }


def confounding_diagnostics(original_df: pd.DataFrame, confounded_df: pd.DataFrame, confounder: str = CONFOUNDER_FEATURE) -> dict:
    orig_smd = _smd(original_df, confounder)
    conf_smd = _smd(confounded_df, confounder)
    return {
        "confounder": confounder,
        "original_n": len(original_df),
        "confounded_n": len(confounded_df),
        "original_treatment_ratio": float(original_df[TREATMENT_COL].mean()),
        "confounded_treatment_ratio": float(confounded_df[TREATMENT_COL].mean()),
        "original_smd_confounder": orig_smd,
        "confounded_smd_confounder": conf_smd,
    }


def _smd(df: pd.DataFrame, col: str) -> float:
    t = df.loc[df[TREATMENT_COL] == 1, col]
    c = df.loc[df[TREATMENT_COL] == 0, col]
    pooled_std = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    return float((t.mean() - c.mean()) / pooled_std) if pooled_std > 0 else float("nan")
