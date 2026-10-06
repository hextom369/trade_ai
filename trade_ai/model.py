"""Models and purged walk-forward training.

Walk-forward: the model is periodically refit on past data only and used to
predict the next block. A gap of ``horizon`` bars is purged between the end of
the training window and the first prediction, so overlapping forward-return
labels cannot leak future information into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class EnsembleClassifier:
    """Average of gradient boosting and regularised logistic regression probabilities."""

    def __init__(self, seed: int = 0):
        self.models = [_make_hgb(seed), _make_logreg()]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def _make_hgb(seed: int = 0):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


def _make_logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=1000))


def make_model(kind: str = "ensemble", seed: int = 0):
    if kind == "hgb":
        return _make_hgb(seed)
    if kind == "logreg":
        return _make_logreg()
    if kind == "ensemble":
        return EnsembleClassifier(seed)
    raise ValueError(f"unknown model kind: {kind}")


@dataclass
class WalkForwardConfig:
    horizon: int = 5          # label horizon in bars (also the purge gap)
    min_train: int = 500      # bars required before the first fit
    retrain_every: int = 60   # refit cadence in bars
    train_window: int | None = None  # None = expanding window, else rolling
    model: str = "ensemble"
    seed: int = 0


def walk_forward_predict(features: pd.DataFrame, labels: pd.Series,
                         cfg: WalkForwardConfig | None = None) -> pd.Series:
    """Return out-of-sample P(label=1) for each row (NaN before the first fit)."""
    cfg = cfg or WalkForwardConfig()
    X = features.to_numpy(dtype=float)
    y = labels.reindex(features.index).to_numpy(dtype=float)
    n = len(features)
    proba = np.full(n, np.nan)

    start = cfg.min_train + cfg.horizon
    for pred_start in range(start, n, cfg.retrain_every):
        pred_end = min(pred_start + cfg.retrain_every, n)
        train_end = pred_start - cfg.horizon  # purge overlapping labels
        train_start = 0 if cfg.train_window is None else max(0, train_end - cfg.train_window)
        Xtr, ytr = X[train_start:train_end], y[train_start:train_end]
        mask = ~np.isnan(ytr) & ~np.all(np.isnan(Xtr), axis=1)
        Xtr, ytr = Xtr[mask], ytr[mask]
        if len(ytr) < 50 or len(np.unique(ytr)) < 2:
            continue
        model = make_model(cfg.model, cfg.seed).fit(Xtr, ytr.astype(int))
        proba[pred_start:pred_end] = model.predict_proba(X[pred_start:pred_end])[:, 1]

    return pd.Series(proba, index=features.index, name="proba")
