"""ATE estimators: Difference in Means, Regression Adjustment (G-computation),
IPW (Horvitz-Thompson), and Doubly Robust / AIPW.

Statistical approach for uncertainty
-------------------------------------
With ~14M rows, bootstrapping is unnecessarily expensive. Every estimator
below is written (or re-derived) as a sample average of a per-unit
"pseudo-outcome" phi_i, so its variance is estimated analytically via the
standard M-estimator / influence-function sandwich:

    ATE_hat = mean(phi_i)
    Var(ATE_hat) = Var(phi_i) / n           (i.e. the usual CLT variance
                                              of a sample mean)

This is *exact* (no approximation) for Difference-in-Means and for
Horvitz-Thompson IPW given fixed weights, and is the textbook
semiparametric-efficient variance for AIPW (Robins, Rotnitzky & Zhao,
1994) when the nuisance models converge fast enough (a standard,
well-justified simplification given the scale here -- see README
"Limitations"). For plug-in Regression Adjustment we report the
plug-in-conditional variance, which slightly understates true uncertainty
because it ignores first-stage model-fitting noise; this is flagged
explicitly in the estimator-comparison table rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURE_COLS, RANDOM_SEED, TREATMENT_COL

Z_95 = 1.959963984540054


def _logreg():
    """Standardized-features logistic regression. The 12 Criteo features
    are on very different scales, which was causing lbfgs to hit the
    iteration cap before converging on some folds (observed for the
    `visit` outcome model); scaling fixes that without changing the
    estimator's statistical target."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


@dataclass
class ATEResult:
    method: str
    ate: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "ate": self.ate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_width": self.ci_high - self.ci_low,
            "n": self.n,
            **self.extra,
        }


def _ci_from_phi(phi: np.ndarray, method: str, extra: dict | None = None) -> ATEResult:
    n = len(phi)
    ate = float(phi.mean())
    var = float(phi.var(ddof=1) / n)
    se = float(np.sqrt(var))
    return ATEResult(
        method=method,
        ate=ate,
        se=se,
        ci_low=ate - Z_95 * se,
        ci_high=ate + Z_95 * se,
        n=n,
        extra=extra or {},
    )


def difference_in_means(df: pd.DataFrame, outcome_col: str, treatment_col: str = TREATMENT_COL) -> ATEResult:
    t = df.loc[df[treatment_col] == 1, outcome_col].to_numpy()
    c = df.loc[df[treatment_col] == 0, outcome_col].to_numpy()
    ate = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return ATEResult(
        method="Difference in Means",
        ate=float(ate),
        se=float(se),
        ci_low=float(ate - Z_95 * se),
        ci_high=float(ate + Z_95 * se),
        n=len(df),
        extra={"n_treated": len(t), "n_control": len(c)},
    )


def cross_fitted_propensity(
    df: pd.DataFrame,
    feature_cols=FEATURE_COLS,
    treatment_col: str = TREATMENT_COL,
    n_splits: int = 3,
    seed: int = RANDOM_SEED,
    clip: tuple[float, float] = (0.01, 0.99),
) -> np.ndarray:
    """Out-of-fold propensity scores for the FULL dataset (avoids using an
    observation's own label to predict its own propensity, which would
    otherwise slightly bias IPW/AIPW toward the training data)."""
    X = df[feature_cols].to_numpy()
    y = df[treatment_col].to_numpy()
    e = np.empty(len(df), dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        model = _logreg()
        model.fit(X[train_idx], y[train_idx])
        e[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return np.clip(e, clip[0], clip[1])


def cross_fitted_outcome_models(
    df: pd.DataFrame,
    outcome_col: str,
    feature_cols=FEATURE_COLS,
    treatment_col: str = TREATMENT_COL,
    n_splits: int = 3,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted mu1(x)=E[Y|X,T=1], mu0(x)=E[Y|X,T=0] for every row,
    fit as two separate logistic-regression outcome models per fold (fit
    on the training fold's treated/control rows respectively, predicted
    out-of-fold for every row in the held-out fold, regardless of that
    row's own observed arm)."""
    X = df[feature_cols].to_numpy()
    y = df[outcome_col].to_numpy()
    t = df[treatment_col].to_numpy()
    mu1 = np.empty(len(df), dtype=np.float64)
    mu0 = np.empty(len(df), dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        tr_t = train_idx[t[train_idx] == 1]
        tr_c = train_idx[t[train_idx] == 0]
        m1 = _logreg().fit(X[tr_t], y[tr_t])
        m0 = _logreg().fit(X[tr_c], y[tr_c])
        mu1[test_idx] = m1.predict_proba(X[test_idx])[:, 1]
        mu0[test_idx] = m0.predict_proba(X[test_idx])[:, 1]
    return mu1, mu0


def regression_adjustment(df: pd.DataFrame, outcome_col: str, mu1: np.ndarray, mu0: np.ndarray) -> ATEResult:
    """G-computation: ATE = E[mu1(X) - mu0(X)] over the full population."""
    phi = mu1 - mu0
    return _ci_from_phi(phi, "Regression Adjustment", extra={
        "note": "plug-in variance; ignores first-stage nuisance-model estimation noise",
    })


def ipw(df: pd.DataFrame, outcome_col: str, treatment_col: str, propensity: np.ndarray) -> ATEResult:
    """Horvitz-Thompson IPW estimator with an influence-function SE."""
    y = df[outcome_col].to_numpy()
    t = df[treatment_col].to_numpy()
    phi = t * y / propensity - (1 - t) * y / (1 - propensity)
    weights_t = 1 / propensity[t == 1]
    weights_c = 1 / (1 - propensity[t == 0])
    ess_t = weights_t.sum() ** 2 / (weights_t ** 2).sum()
    ess_c = weights_c.sum() ** 2 / (weights_c ** 2).sum()
    return _ci_from_phi(phi, "IPW (Horvitz-Thompson)", extra={
        "max_weight_treated": float(weights_t.max()),
        "max_weight_control": float(weights_c.max()),
        "effective_n_treated": float(ess_t),
        "effective_n_control": float(ess_c),
    })


def ipw_stabilized(df: pd.DataFrame, outcome_col: str, treatment_col: str, propensity: np.ndarray) -> ATEResult:
    """Hajek IPW with an influence-function SE conditional on fitted propensities."""
    y = df[outcome_col].to_numpy()
    t = df[treatment_col].to_numpy()
    w1 = t / propensity
    w0 = (1 - t) / (1 - propensity)
    mu1_hat = (w1 * y).sum() / w1.sum()
    mu0_hat = (w0 * y).sum() / w0.sum()
    ate = mu1_hat - mu0_hat
    influence = (
        w1 * (y - mu1_hat) / w1.mean()
        - w0 * (y - mu0_hat) / w0.mean()
    )
    return _ci_from_phi(
        ate + influence,
        "IPW (Hajek, stabilized)",
        extra={"note": "IF variance conditional on fitted propensity scores"},
    )


def doubly_robust(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    propensity: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
) -> ATEResult:
    """Augmented IPW (AIPW): consistent if EITHER the propensity model OR
    the outcome model is correctly specified -- the estimator's namesake
    "double robustness"."""
    y = df[outcome_col].to_numpy()
    t = df[treatment_col].to_numpy()
    phi = (
        (mu1 - mu0)
        + t * (y - mu1) / propensity
        - (1 - t) * (y - mu0) / (1 - propensity)
    )
    return _ci_from_phi(phi, "Doubly Robust (AIPW)")


def run_all_estimators(
    df: pd.DataFrame,
    outcome_col: str,
    feature_cols=FEATURE_COLS,
    treatment_col: str = TREATMENT_COL,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper: fits shared nuisance models once, then computes
    all four estimators + the stabilized-IPW robustness check."""
    propensity = cross_fitted_propensity(df, feature_cols, treatment_col, seed=seed)
    mu1, mu0 = cross_fitted_outcome_models(df, outcome_col, feature_cols, treatment_col, seed=seed)

    primary_results = [
        difference_in_means(df, outcome_col, treatment_col),
        regression_adjustment(df, outcome_col, mu1, mu0),
        ipw(df, outcome_col, treatment_col, propensity),
        doubly_robust(df, outcome_col, treatment_col, propensity, mu1, mu0),
    ]
    stabilized = ipw_stabilized(df, outcome_col, treatment_col, propensity)
    table = pd.DataFrame([r.as_row() for r in [*primary_results, stabilized]])
    nuisances = {"propensity": propensity, "mu1": mu1, "mu0": mu0, "stabilized_ipw": stabilized}
    return table, nuisances
