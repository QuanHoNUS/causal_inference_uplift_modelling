"""Thin regressor-interface wrapper around a classifier.

EconML's meta-learners (S/T/X-Learner) expect a `.fit(X, y).predict(X)`
"regression" interface and use the raw `.predict(X)` output as the
estimated potential-outcome / effect surface. For a binary outcome
(conversion), we want that surface to be the predicted PROBABILITY of
conversion, not a hard 0/1 class label -- so we wrap the classifier and
route `.predict` through `.predict_proba(X)[:, 1]`.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin, clone


class ClassifierAsRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y, **kwargs):
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        proba = self.classifier_.predict_proba(X)
        return proba[:, 1]
