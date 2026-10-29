"""Targeting-policy simulation and offline evaluation.

Offline policy value estimation
--------------------------------
We cannot observe both potential outcomes for any customer, so a policy
that decides "treat" or "don't treat" from X alone cannot simply be
"applied" to historical data and have its outcome read off directly
(e.g. for a treat-none policy we never observe what an actually-treated
customer would have done under no treatment). Because the ORIGINAL
treatment T in this dataset was assigned independently of X with known
probability e (empirically ~0.85, verified in the treatment-assignment
analysis), we can use the standard Inverse Propensity Scoring (IPS)
off-policy value estimator (Zhao et al., 2017; Athey & Wager, 2021):

    V_hat(policy) = mean_i[ 1{policy(X_i)=1} * T_i * Y_i / e
                           + 1{policy(X_i)=0} * (1-T_i) * Y_i / (1-e) ]

This is unbiased for "the expected outcome rate if the policy's decisions
had been the actual treatment assignment", *because* the policy decision
depends only on X (via the uplift score) and X is independent of (T, Y)
given randomization -- exactly the setting this dataset provides. It
reduces correctly to the naive treated/control rates for the treat-all /
treat-none policies, which is a useful internal-consistency check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054


def _policy_value_contributions(
    y: np.ndarray,
    t: np.ndarray,
    policy_mask: np.ndarray,
    e_treatment: float,
) -> np.ndarray:
    treated_contrib = np.where(t == 1, y / e_treatment, 0.0)
    control_contrib = np.where(t == 0, y / (1 - e_treatment), 0.0)
    return np.where(policy_mask, treated_contrib, control_contrib)


def offline_policy_value(y: np.ndarray, t: np.ndarray, policy_mask: np.ndarray, e_treatment: float) -> float:
    return float(_policy_value_contributions(y, t, policy_mask, e_treatment).mean())


def top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    """Apply a pre-specified top-fraction policy without looking at outcomes."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    n_targeted = int(np.ceil(fraction * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    if n_targeted:
        order = np.argsort(-score, kind="stable")
        mask[order[:n_targeted]] = True
    return mask


def build_policies(score: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    n = len(score)
    rng = np.random.default_rng(seed)

    policies = {
        "Treat All": np.ones(n, dtype=bool),
        "Treat None": np.zeros(n, dtype=bool),
        "Random 50%": rng.random(n) < 0.5,
        "Positive Uplift Only": score > 0,
        "Top 10% Uplift": top_fraction_mask(score, 0.10),
        "Top 20% Uplift": top_fraction_mask(score, 0.20),
    }
    return policies


def policy_profit_curve(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    e_treatment: float,
    value_per_conversion: float,
    cost_per_contact: float,
    candidate_fractions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate a pre-specified grid of top-fraction policies with IPS.

    The curve is suitable for *selection* on a validation set or for
    *evaluation* on an independent test set. It must not be optimized and
    reported on the same observations.
    """
    if candidate_fractions is None:
        candidate_fractions = np.linspace(0.0, 1.0, 101)
    fractions = np.asarray(candidate_fractions, dtype=float)
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("candidate_fractions must be between 0 and 1")

    order = np.argsort(-score, kind="stable")
    y_sorted = np.asarray(y)[order]
    t_sorted = np.asarray(t)[order]
    incremental_contrib = np.where(
        t_sorted == 1,
        y_sorted / e_treatment,
        -y_sorted / (1 - e_treatment),
    )
    cumulative_incremental = np.concatenate(([0.0], np.cumsum(incremental_contrib)))
    n = len(y_sorted)
    rows = []
    for fraction in fractions:
        n_targeted = min(int(np.ceil(fraction * n)), n)
        incremental_conversions = float(cumulative_incremental[n_targeted])
        cost = n_targeted * cost_per_contact
        revenue = incremental_conversions * value_per_conversion
        rows.append({
            "population_fraction": n_targeted / n,
            "n_targeted": n_targeted,
            "incremental_conversions": incremental_conversions,
            "cost": cost,
            "revenue": revenue,
            "profit": revenue - cost,
        })
    return pd.DataFrame(rows).drop_duplicates("n_targeted").reset_index(drop=True)


def find_profit_maximizing_threshold(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    e_treatment: float,
    value_per_conversion: float,
    cost_per_contact: float,
    candidate_fractions: np.ndarray | None = None,
) -> dict:
    """Select a top-fraction policy on a validation set.

    The default 1-percentage-point grid is declared in advance to reduce
    noise from searching every possible row-level cutoff. The returned
    fraction must be applied to an independent test set for final reporting.
    """
    df = policy_profit_curve(
        y,
        t,
        score,
        e_treatment,
        value_per_conversion,
        cost_per_contact,
        candidate_fractions,
    )
    best_row = df.loc[df["profit"].idxmax()]
    k_star_fraction = float(best_row["population_fraction"])
    return {
        "k_star_fraction": k_star_fraction,
        "k_star_customers": int(best_row["n_targeted"]),
        "expected_incremental_conversions": float(best_row["incremental_conversions"]),
        "expected_cost": float(best_row["cost"]),
        "expected_revenue": float(best_row["revenue"]),
        "expected_profit": float(best_row["profit"]),
        "mask": top_fraction_mask(score, k_star_fraction),
        "curve": df,
    }


def evaluate_policies(
    y: np.ndarray,
    t: np.ndarray,
    policies: dict[str, np.ndarray],
    e_treatment: float,
    value_per_conversion: float,
    cost_per_contact: float,
) -> pd.DataFrame:
    n = len(y)
    rows = []
    baseline_mask = np.zeros(n, dtype=bool)
    baseline_contrib = _policy_value_contributions(y, t, baseline_mask, e_treatment)
    baseline_conversions = baseline_contrib.mean() * n
    for name, mask in policies.items():
        coverage = float(mask.mean())
        policy_contrib = _policy_value_contributions(y, t, mask, e_treatment)
        exp_rate = float(policy_contrib.mean())
        exp_conversions = exp_rate * n
        incremental_contrib = policy_contrib - baseline_contrib
        incremental_conversions = float(incremental_contrib.mean() * n)
        incremental_conversions_se = float(incremental_contrib.std(ddof=1) * np.sqrt(n))
        cost = coverage * n * cost_per_contact
        revenue = incremental_conversions * value_per_conversion
        profit = revenue - cost
        profit_se = incremental_conversions_se * value_per_conversion
        rows.append({
            "policy": name,
            "coverage": coverage,
            "n_targeted": int(mask.sum()),
            "expected_conversion_rate": exp_rate,
            "expected_conversions": exp_conversions,
            "incremental_conversions_vs_treat_none": incremental_conversions,
            "incremental_conversions_se": incremental_conversions_se,
            "campaign_cost": cost,
            "incremental_revenue": revenue,
            "net_profit": profit,
            "net_profit_ci_low": profit - Z_95 * profit_se,
            "net_profit_ci_high": profit + Z_95 * profit_se,
            "profit_per_contact": profit / mask.sum() if mask.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("net_profit", ascending=False).reset_index(drop=True)


def cost_sensitivity_sweep(
    selection_y: np.ndarray,
    selection_t: np.ndarray,
    selection_score: np.ndarray,
    evaluation_y: np.ndarray,
    evaluation_t: np.ndarray,
    evaluation_score: np.ndarray,
    e_treatment: float,
    value_per_conversion_grid: list[float],
    cost_per_contact_grid: list[float],
) -> pd.DataFrame:
    rows = []
    for v in value_per_conversion_grid:
        for c in cost_per_contact_grid:
            selected = find_profit_maximizing_threshold(
                selection_y,
                selection_t,
                selection_score,
                e_treatment,
                v,
                c,
            )
            evaluation_mask = top_fraction_mask(evaluation_score, selected["k_star_fraction"])
            evaluated = evaluate_policies(
                evaluation_y,
                evaluation_t,
                {"Selected Top-K": evaluation_mask},
                e_treatment,
                v,
                c,
            ).iloc[0]
            rows.append({
                "value_per_conversion": v,
                "cost_per_contact": c,
                "selected_coverage_validation": selected["k_star_fraction"],
                "selection_profit_validation": selected["expected_profit"],
                "evaluation_profit_test": evaluated["net_profit"],
                "evaluation_profit_ci_low_test": evaluated["net_profit_ci_low"],
                "evaluation_profit_ci_high_test": evaluated["net_profit_ci_high"],
            })
    return pd.DataFrame(rows)
