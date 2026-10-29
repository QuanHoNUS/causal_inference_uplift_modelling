"""Uplift-model evaluation: Qini curve/coefficient, AUUC (incremental-gain
curve/coefficient), calibration by decile, and average uplift.

All metrics are implemented from first principles (not delegated to a
library black box) so the report can explain exactly what each one
measures. Definitions follow Radcliffe (2007) and the standard treatment in
Gutierrez & Gerardy (2017) "Causal Inference and Uplift Modelling: A
Review".

Notation, for a test set ranked by descending predicted uplift score and a
"top-k" cutoff:
  n1k, n0k = number of treated / control units in the top-k
  r1k, r0k = sum of observed outcome among treated / control in the top-k
  n1, n0   = TOTAL treated / control units in the whole test set

Qini curve:            Q(k) = r1k - r0k * (n1 / n0)
    "cumulative incremental responders", rescaling the control group's
    cumulative response so it's directly comparable to the treated group's
    (adjusts for n1 != n0, which is the case here: 85% treated / 15%
    control).

Incremental Gain curve: G(k) = (r1k/n1k - r0k/n0k) * (n1k + n0k)
    "if the top-k population were entirely treated, how many incremental
    conversions would we expect", extrapolating the observed per-capita
    treated-vs-control gap within the top-k to the whole top-k group.

Both curves are compared to the RANDOM-TARGETING baseline, the straight
line from (0, 0) to (1, curve value at k=100%) -- what you'd get by
selecting customers in a random order rather than by predicted uplift. The
Qini coefficient / AUUC score is the area between the model's curve and
that random baseline (positive = model beats random targeting).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import confint_proportions_2indep

from src.utils.config import RANDOM_SEED

NORMALIZED_METRIC_SCALE = 100_000


def _cumulative_arrays(y: np.ndarray, t: np.ndarray, score: np.ndarray):
    order = np.argsort(-score, kind="stable")
    y_s, t_s = y[order], t[order]
    cum_n1 = np.cumsum(t_s)
    cum_n0 = np.cumsum(1 - t_s)
    cum_r1 = np.cumsum(y_s * t_s)
    cum_r0 = np.cumsum(y_s * (1 - t_s))
    n = len(y)
    pop_frac = np.arange(1, n + 1) / n
    return pop_frac, cum_n1, cum_n0, cum_r1, cum_r0


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    pop_frac, cum_n1, cum_n0, cum_r1, cum_r0 = _cumulative_arrays(y, t, score)
    n1_total, n0_total = cum_n1[-1], cum_n0[-1]
    ratio = n1_total / n0_total if n0_total > 0 else 0.0
    qini = cum_r1 - cum_r0 * ratio
    random_qini = pop_frac * qini[-1]
    return pd.DataFrame({
        "population_fraction": pop_frac,
        "qini": qini,
        "random_qini": random_qini,
    })


def incremental_gain_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    pop_frac, cum_n1, cum_n0, cum_r1, cum_r0 = _cumulative_arrays(y, t, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate1 = np.where(cum_n1 > 0, cum_r1 / np.maximum(cum_n1, 1), 0.0)
        rate0 = np.where(cum_n0 > 0, cum_r0 / np.maximum(cum_n0, 1), 0.0)
    gain = (rate1 - rate0) * (cum_n1 + cum_n0)
    random_gain = pop_frac * gain[-1]
    return pd.DataFrame({
        "population_fraction": pop_frac,
        "incremental_gain": gain,
        "random_gain": random_gain,
    })


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    df = qini_curve(y, t, score)
    area_model = np.trapz(df["qini"], df["population_fraction"])
    area_random = np.trapz(df["random_qini"], df["population_fraction"])
    return float(area_model - area_random)


def auuc_score(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    df = incremental_gain_curve(y, t, score)
    area_model = np.trapz(df["incremental_gain"], df["population_fraction"])
    area_random = np.trapz(df["random_gain"], df["population_fraction"])
    return float(area_model - area_random)


def average_uplift(score: np.ndarray) -> float:
    return float(np.mean(score))


def _difference_in_proportions_ci(
    treated_outcomes: pd.Series,
    control_outcomes: pd.Series,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(treated_outcomes) == 0 or len(control_outcomes) == 0:
        return float("nan"), float("nan")
    return tuple(float(value) for value in confint_proportions_2indep(
        count1=int(treated_outcomes.sum()),
        nobs1=len(treated_outcomes),
        count2=int(control_outcomes.sum()),
        nobs2=len(control_outcomes),
        method="newcomb",
        compare="diff",
        alpha=alpha,
    ))


def calibration_by_decile(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    decile = pd.qcut(score, n_deciles, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "t": t, "score": score, "decile": decile})
    rows = []
    for d, g in df.groupby("decile"):
        treated = g.loc[g.t == 1, "y"]
        control = g.loc[g.t == 0, "y"]
        ci_low, ci_high = _difference_in_proportions_ci(treated, control)
        rows.append({
            "decile": int(d),
            "n": len(g),
            "n_treated": len(treated),
            "n_control": len(control),
            "mean_predicted_uplift": g["score"].mean(),
            "actual_treated_rate": treated.mean() if len(treated) else np.nan,
            "actual_control_rate": control.mean() if len(control) else np.nan,
            "actual_uplift_ci_low": ci_low,
            "actual_uplift_ci_high": ci_high,
        })
    out = pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
    out["actual_uplift"] = out["actual_treated_rate"] - out["actual_control_rate"]
    # Decile 0 = lowest predicted uplift (qcut ascending); flip so decile 1 = TOP predicted uplift, matching how targeting policies rank customers.
    out["decile_from_top"] = out["decile"].max() - out["decile"] + 1
    return out.sort_values("decile_from_top").reset_index(drop=True)


def evaluate_all_models(y: np.ndarray, t: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, score in scores.items():
        qini = qini_coefficient(y, t, score)
        auuc = auuc_score(y, t, score)
        rows.append({
            "model": name,
            "qini_coefficient": qini,
            "auuc_score": auuc,
            "qini_per_100k": qini / len(y) * NORMALIZED_METRIC_SCALE,
            "auuc_per_100k": auuc / len(y) * NORMALIZED_METRIC_SCALE,
            "average_predicted_uplift": average_uplift(score),
        })
    return pd.DataFrame(rows).sort_values("qini_coefficient", ascending=False).reset_index(drop=True)


def _qini_auuc_from_ranked(y_ranked: np.ndarray, t_ranked: np.ndarray) -> tuple[float, float]:
    """Compute Qini and AUUC after observations have been ranked by score."""
    n = len(y_ranked)
    pop_frac = np.arange(1, n + 1) / n
    cum_n1 = np.cumsum(t_ranked)
    cum_n0 = np.cumsum(1 - t_ranked)
    cum_r1 = np.cumsum(y_ranked * t_ranked)
    cum_r0 = np.cumsum(y_ranked * (1 - t_ranked))

    ratio = cum_n1[-1] / cum_n0[-1]
    qini = cum_r1 - cum_r0 * ratio
    qini_value = np.trapz(qini - pop_frac * qini[-1], pop_frac)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate1 = np.where(cum_n1 > 0, cum_r1 / np.maximum(cum_n1, 1), 0.0)
        rate0 = np.where(cum_n0 > 0, cum_r0 / np.maximum(cum_n0, 1), 0.0)
    gain = (rate1 - rate0) * (cum_n1 + cum_n0)
    auuc_value = np.trapz(gain - pop_frac * gain[-1], pop_frac)
    return float(qini_value), float(auuc_value)


def bootstrap_uplift_metrics(
    y: np.ndarray,
    t: np.ndarray,
    scores: dict[str, np.ndarray],
    n_bootstrap: int = 200,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired, treatment-stratified bootstrap intervals for Qini and AUUC.

    The same bootstrap rows are used for every model, so pairwise model
    differences retain their covariance. Metrics are normalized to expected
    incremental conversions per 100,000 customers; raw Qini/AUUC values scale
    with evaluation-set size and should only be compared on the same sample.
    """
    y = np.asarray(y)
    t = np.asarray(t)
    model_names = list(scores)
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if any(len(score) != len(y) for score in scores.values()):
        raise ValueError("all score arrays must have the same length as y")

    treated_idx = np.flatnonzero(t == 1)
    control_idx = np.flatnonzero(t == 0)
    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("bootstrap requires both treatment arms")

    point_table = evaluate_all_models(y, t, scores).set_index("model")
    qini_draws = np.empty((n_bootstrap, len(model_names)), dtype=float)
    auuc_draws = np.empty_like(qini_draws)
    rng = np.random.default_rng(seed)

    for bootstrap_idx in range(n_bootstrap):
        sampled_idx = np.concatenate((
            rng.choice(treated_idx, size=len(treated_idx), replace=True),
            rng.choice(control_idx, size=len(control_idx), replace=True),
        ))
        y_boot = y[sampled_idx]
        t_boot = t[sampled_idx]
        for model_idx, name in enumerate(model_names):
            score_boot = np.asarray(scores[name])[sampled_idx]
            order = np.argsort(-score_boot, kind="stable")
            qini, auuc = _qini_auuc_from_ranked(y_boot[order], t_boot[order])
            qini_draws[bootstrap_idx, model_idx] = qini / len(y_boot) * NORMALIZED_METRIC_SCALE
            auuc_draws[bootstrap_idx, model_idx] = auuc / len(y_boot) * NORMALIZED_METRIC_SCALE

    rows = []
    for model_idx, name in enumerate(model_names):
        point = point_table.loc[name]
        rows.append({
            "model": name,
            "qini_coefficient": point["qini_coefficient"],
            "qini_per_100k": point["qini_per_100k"],
            "qini_ci_low_per_100k": np.quantile(qini_draws[:, model_idx], 0.025),
            "qini_ci_high_per_100k": np.quantile(qini_draws[:, model_idx], 0.975),
            "auuc_score": point["auuc_score"],
            "auuc_per_100k": point["auuc_per_100k"],
            "auuc_ci_low_per_100k": np.quantile(auuc_draws[:, model_idx], 0.025),
            "auuc_ci_high_per_100k": np.quantile(auuc_draws[:, model_idx], 0.975),
            "bootstrap_repetitions": n_bootstrap,
        })
    summary = pd.DataFrame(rows).sort_values("qini_per_100k", ascending=False).reset_index(drop=True)

    pairwise_rows = []
    for first_idx, first_name in enumerate(model_names):
        for second_idx in range(first_idx + 1, len(model_names)):
            second_name = model_names[second_idx]
            qini_diff = qini_draws[:, first_idx] - qini_draws[:, second_idx]
            auuc_diff = auuc_draws[:, first_idx] - auuc_draws[:, second_idx]
            pairwise_rows.append({
                "model_a": first_name,
                "model_b": second_name,
                "qini_difference_per_100k": (
                    point_table.loc[first_name, "qini_per_100k"]
                    - point_table.loc[second_name, "qini_per_100k"]
                ),
                "qini_difference_ci_low": np.quantile(qini_diff, 0.025),
                "qini_difference_ci_high": np.quantile(qini_diff, 0.975),
                "probability_qini_a_gt_b": np.mean(qini_diff > 0),
                "auuc_difference_per_100k": (
                    point_table.loc[first_name, "auuc_per_100k"]
                    - point_table.loc[second_name, "auuc_per_100k"]
                ),
                "auuc_difference_ci_low": np.quantile(auuc_diff, 0.025),
                "auuc_difference_ci_high": np.quantile(auuc_diff, 0.975),
                "probability_auuc_a_gt_b": np.mean(auuc_diff > 0),
                "bootstrap_repetitions": n_bootstrap,
            })
    pairwise = pd.DataFrame(pairwise_rows)
    return summary, pairwise


def _downsample_for_plot(df: pd.DataFrame, max_points: int = 800) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx]


def plot_qini_curves(y, t, scores: dict, save_path=None):
    from src.visualization.style import MODEL_COLORS

    fig, ax = plt.subplots(figsize=(8, 6))
    random_plotted = False
    for name, score in scores.items():
        df = _downsample_for_plot(qini_curve(y, t, score))
        color = MODEL_COLORS.get(name, None)
        ax.plot(df["population_fraction"], df["qini"], label=name, color=color, linewidth=2)
        if not random_plotted:
            ax.plot(df["population_fraction"], df["random_qini"], label="Random targeting", color=MODEL_COLORS["Random"], linestyle="--", linewidth=1.5)
            random_plotted = True
    ax.set_xlabel("Fraction of customers targeted (ranked by predicted uplift)")
    ax.set_ylabel("Cumulative incremental conversions (Qini)")
    ax.set_title("Qini Curve: Model Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_gain_curves(y, t, scores: dict, save_path=None):
    from src.visualization.style import MODEL_COLORS

    fig, ax = plt.subplots(figsize=(8, 6))
    random_plotted = False
    for name, score in scores.items():
        df = _downsample_for_plot(incremental_gain_curve(y, t, score))
        color = MODEL_COLORS.get(name, None)
        ax.plot(df["population_fraction"], df["incremental_gain"], label=name, color=color, linewidth=2)
        if not random_plotted:
            ax.plot(df["population_fraction"], df["random_gain"], label="Random targeting", color=MODEL_COLORS["Random"], linestyle="--", linewidth=1.5)
            random_plotted = True
    ax.set_xlabel("Fraction of customers targeted (ranked by predicted uplift)")
    ax.set_ylabel("Estimated incremental conversions if top-k% treated")
    ax.set_title("Incremental Gain Curve: Model Comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def segment_profile_by_uplift(
    df: pd.DataFrame,
    score_col: str,
    feature_cols: list[str],
    outcome_col: str,
    treatment_col: str,
    top_frac: float = 0.1,
    bottom_frac: float = 0.1,
) -> pd.DataFrame:
    """Characterize WHO the model thinks benefits most (top decile of
    predicted uplift) vs. who it thinks may be harmed/unaffected (bottom
    decile, i.e. most negative predicted uplift), by comparing feature
    means and the empirical (observed) uplift within each segment.

    Used to answer the assignment's "which segments benefit the most /
    which may be harmed" business question with an actual customer
    profile, not just an aggregate number.
    """
    n = len(df)
    ranked = df.sort_values(score_col, ascending=False)
    top = ranked.iloc[: int(top_frac * n)]
    bottom = ranked.iloc[int((1 - bottom_frac) * n):]

    def _profile(seg: pd.DataFrame, label: str) -> dict:
        t = seg.loc[seg[treatment_col] == 1, outcome_col]
        c = seg.loc[seg[treatment_col] == 0, outcome_col]
        ci_low, ci_high = _difference_in_proportions_ci(t, c)
        row = {
            "segment": label,
            "n": len(seg),
            "n_treated": len(t),
            "n_control": len(c),
            "mean_predicted_uplift": seg[score_col].mean(),
            "actual_treated_rate": t.mean() if len(t) else float("nan"),
            "actual_control_rate": c.mean() if len(c) else float("nan"),
            "actual_uplift_ci_low": ci_low,
            "actual_uplift_ci_high": ci_high,
        }
        row["actual_uplift"] = row["actual_treated_rate"] - row["actual_control_rate"]
        for f in feature_cols:
            row[f"mean_{f}"] = seg[f].mean()
        return row

    out = pd.DataFrame([
        _profile(top, f"Top {top_frac:.0%} predicted uplift (target these)"),
        _profile(bottom, f"Bottom {bottom_frac:.0%} predicted uplift (avoid targeting)"),
    ])
    return out


def plot_calibration(calib_table: pd.DataFrame, model_name: str, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = calib_table["decile_from_top"]
    width = 0.38
    ax.bar(x - width / 2, calib_table["mean_predicted_uplift"], width=width, label="Predicted uplift", color="#2563eb")
    actual = calib_table["actual_uplift"]
    error = np.vstack((
        actual - calib_table["actual_uplift_ci_low"],
        calib_table["actual_uplift_ci_high"] - actual,
    ))
    ax.bar(
        x + width / 2,
        actual,
        width=width,
        yerr=error,
        capsize=2.5,
        label="Actual uplift (95% CI)",
        color="#16a34a",
    )
    ax.set_xlabel("Decile (1 = highest predicted uplift)")
    ax.set_ylabel("Uplift (P(Y=1|T=1) - P(Y=1|T=0))")
    ax.set_title(f"Calibration by Decile: {model_name}")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
