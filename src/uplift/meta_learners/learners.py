"""S / T / X / DR meta-learners for CATE estimation, built on EconML with
LightGBM base learners (fast on millions of rows, matches the assignment's
recommended-libraries list).

All four learners share the same base-learner recipe so differences in
their Qini/AUUC performance (Section 7.4 ablation) reflect the *meta-learner
architecture*, not incidental base-learner tuning differences.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner

from src.uplift.meta_learners.wrappers import ClassifierAsRegressor

LGBM_PARAMS = dict(n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=31, verbosity=-1, n_jobs=-1)


def _base_classifier():
    return LGBMClassifier(**LGBM_PARAMS)


def _base_regressor_wrapped():
    """Predicts the binary outcome's probability -- used for the first-stage
    "potential outcome" models (mu0, mu1) in every meta-learner."""
    return ClassifierAsRegressor(_base_classifier())


def _base_regressor():
    """A genuine regressor (not a wrapped classifier) -- needed for
    X-Learner's second stage, which regresses a CONTINUOUS imputed
    treatment-effect target (D = Y - mu0hat(X) for treated units,
    mu1hat(X) - Y for control units), not the original binary Y."""
    return LGBMRegressor(**LGBM_PARAMS)


def _propensity_model():
    """Standardized-features logistic regression -- see
    src/causal/ate/estimators.py for why scaling+more iterations matters
    (the 12 Criteo features are on very different scales)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def fit_s_learner(X, T, Y):
    model = SLearner(overall_model=_base_regressor_wrapped())
    model.fit(Y, T, X=X)
    return model


def fit_t_learner(X, T, Y):
    model = TLearner(models=_base_regressor_wrapped())
    model.fit(Y, T, X=X)
    return model


def fit_x_learner(X, T, Y):
    model = XLearner(
        models=_base_regressor_wrapped(),
        cate_models=_base_regressor(),
        propensity_model=_propensity_model(),
    )
    model.fit(Y, T, X=X)
    return model


def fit_dr_learner(X, T, Y, cv: int = 3):
    model = DRLearner(
        model_propensity=_propensity_model(),
        model_regression=_base_classifier(),
        discrete_outcome=True,
        cv=cv,
        random_state=42,
    )
    model.fit(Y, T, X=X)
    return model


FITTERS = {
    "S-Learner": fit_s_learner,
    "T-Learner": fit_t_learner,
    "X-Learner": fit_x_learner,
    "DR-Learner": fit_dr_learner,
}
