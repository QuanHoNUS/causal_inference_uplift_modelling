"""Causal Forest (EconML's honest, orthogonalized CausalForestDML) --
the tree-based CATE estimator in the model comparison."""
from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor

from econml.dml import CausalForestDML


def fit_causal_forest(X, T, Y, n_estimators: int = 300, subforest_size: int = 4, random_state: int = 42):
    # EconML's GRF backend requires n_estimators divisible by subforest_size
    # (it builds n_estimators/subforest_size independent "subforests" for its
    # honesty-splitting variance estimator) -- round up rather than error.
    if n_estimators % subforest_size != 0:
        n_estimators = ((n_estimators // subforest_size) + 1) * subforest_size

    model = CausalForestDML(
        model_y=LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, verbosity=-1, n_jobs=-1),
        model_t=LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, verbosity=-1, n_jobs=-1),
        discrete_outcome=True,
        discrete_treatment=True,
        n_estimators=n_estimators,
        subforest_size=subforest_size,
        min_samples_leaf=50,
        max_depth=None,
        cv=3,
        honest=True,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(Y, T, X=X)
    return model
