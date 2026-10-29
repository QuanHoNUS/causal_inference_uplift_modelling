"""Fast, data-free smoke tests: importability + pure-function correctness
on tiny synthetic arrays. These do not touch the real Criteo dataset (too
large for a unit test) -- they guard the statistical formulas themselves.
"""
import numpy as np
import pandas as pd


def test_imports():
    import src.utils.config  # noqa: F401
    import src.preprocessing.load_data  # noqa: F401
    import src.preprocessing.eda  # noqa: F401
    import src.causal.propensity.propensity  # noqa: F401
    import src.causal.ate.estimators  # noqa: F401
    import src.causal.ate.confounding_simulation  # noqa: F401
    import src.causal.dag.dag  # noqa: F401
    import src.evaluation.uplift_metrics  # noqa: F401
    import src.policy.policies  # noqa: F401


def test_difference_in_means_matches_manual_calc():
    from src.causal.ate.estimators import difference_in_means

    df = pd.DataFrame({
        "treatment": [1, 1, 1, 0, 0, 0],
        "y": [1, 1, 0, 0, 0, 0],
    })
    result = difference_in_means(df, outcome_col="y", treatment_col="treatment")
    assert np.isclose(result.ate, 2 / 3 - 0.0)
    assert result.n == 6


def test_hajek_ipw_has_finite_influence_function_interval():
    from src.causal.ate.estimators import ipw_stabilized

    rng = np.random.default_rng(23)
    n = 20_000
    treatment = (rng.random(n) < 0.5).astype(int)
    outcome = (rng.random(n) < 0.05 + 0.03 * treatment).astype(int)
    df = pd.DataFrame({"treatment": treatment, "y": outcome})

    result = ipw_stabilized(
        df,
        outcome_col="y",
        treatment_col="treatment",
        propensity=np.full(n, 0.5),
    )

    naive_difference = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    assert np.isclose(result.ate, naive_difference)
    assert np.isfinite(result.se)
    assert result.ci_low < result.ate < result.ci_high


def test_qini_curve_is_zero_for_random_scores_in_expectation():
    from src.evaluation.uplift_metrics import qini_coefficient

    rng = np.random.default_rng(0)
    n = 5000
    t = (rng.random(n) < 0.8).astype(int)
    # Outcome independent of score by construction -> a well-specified model should score ~0 uplift skill.
    y = (rng.random(n) < 0.1 + 0.05 * t).astype(int)
    random_score = rng.random(n)
    q = qini_coefficient(y, t, random_score)
    # Not exactly zero (finite-sample noise) but should be small relative to the achievable range.
    assert abs(q) < 0.05 * n


def test_smd_zero_for_identical_arms():
    from src.causal.propensity.propensity import standardized_mean_difference

    df = pd.DataFrame({
        "treatment": [1, 0] * 500,
        "f0": [1.0, 1.0] * 500,
    })
    out = standardized_mean_difference(df, feature_cols=["f0"])
    assert np.isnan(out.loc[0, "smd"]) or abs(out.loc[0, "smd"]) < 1e-9


def test_offline_policy_value_matches_naive_rates_for_extreme_policies():
    from src.policy.policies import offline_policy_value

    rng = np.random.default_rng(1)
    n = 20000
    e = 0.85
    t = (rng.random(n) < e).astype(int)
    y = (rng.random(n) < np.where(t == 1, 0.05, 0.03)).astype(float)

    treat_all = np.ones(n, dtype=bool)
    treat_none = np.zeros(n, dtype=bool)

    v_all = offline_policy_value(y, t, treat_all, e)
    v_none = offline_policy_value(y, t, treat_none, e)

    naive_treated_rate = y[t == 1].mean()
    naive_control_rate = y[t == 0].mean()

    assert np.isclose(v_all, naive_treated_rate, atol=1e-2)
    assert np.isclose(v_none, naive_control_rate, atol=1e-2)


def test_profit_threshold_is_selected_without_evaluation_outcomes():
    from src.policy.policies import find_profit_maximizing_threshold, top_fraction_mask

    rng = np.random.default_rng(7)
    n = 5000
    score = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.05 + 0.10 * t * (score > 1)).astype(float)

    selected = find_profit_maximizing_threshold(
        y,
        t,
        score,
        e_treatment=0.5,
        value_per_conversion=10.0,
        cost_per_contact=0.1,
    )
    evaluation_score = rng.normal(size=2000)
    evaluation_mask = top_fraction_mask(evaluation_score, selected["k_star_fraction"])

    assert 0 <= selected["k_star_fraction"] <= 1
    assert np.isclose(evaluation_mask.mean(), selected["k_star_fraction"], atol=1 / len(evaluation_score))


def test_target_population_reference_uses_retention_estimand():
    from src.causal.ate.confounding_simulation import target_population_rct_reference

    rng = np.random.default_rng(11)
    n = 100_000
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(int)
    treatment_effect = 0.03 + 0.02 * x
    baseline = 0.10 + 0.03 * x
    y_probability = np.clip(baseline + t * treatment_effect, 0.001, 0.999)
    y = (rng.random(n) < y_probability).astype(int)
    df = pd.DataFrame({"f4": x, "treatment": t, "conversion": y})

    reference = target_population_rct_reference(df, outcome_col="conversion", beta=1.0, z_clip=2.0)

    assert reference["estimand"] == "ATE in the marginal-retention target population"
    assert reference["ci_low"] < reference["ate"] < reference["ci_high"]
    assert not np.isclose(reference["ate"], 0.03, atol=1e-3)


def test_uplift_bootstrap_is_paired_and_reproducible():
    from src.evaluation.uplift_metrics import bootstrap_uplift_metrics

    rng = np.random.default_rng(19)
    n = 2000
    t = (rng.random(n) < 0.5).astype(int)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.10 + 0.08 * t * (x > 0)).astype(int)
    scores = {"good": x, "bad": -x}

    first_summary, first_pairwise = bootstrap_uplift_metrics(y, t, scores, n_bootstrap=20, seed=3)
    second_summary, second_pairwise = bootstrap_uplift_metrics(y, t, scores, n_bootstrap=20, seed=3)

    pd.testing.assert_frame_equal(first_summary, second_summary)
    pd.testing.assert_frame_equal(first_pairwise, second_pairwise)
    comparison = first_pairwise.iloc[0]
    assert comparison["qini_difference_per_100k"] > 0
    assert 0 <= comparison["probability_qini_a_gt_b"] <= 1
