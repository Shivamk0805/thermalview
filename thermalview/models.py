"""The compared forecasters. Each predicts the temperature change over
the horizon; persistence predicts zero change."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Persistence:
    """temp(t+h) = temp(t)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _rf(max_depth, min_samples_leaf, seed):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 max_features=0.5, n_jobs=-1, random_state=seed)


def _hgb(learning_rate, max_leaf_nodes, seed):
    return HistGradientBoostingRegressor(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                         max_iter=400, l2_regularization=1.0, early_stopping=False,
                                         random_state=seed)


def _mlp(alpha, seed):
    """Small feed-forward network on the same windowed features (early stopping on
    an internal 10% split of the training rows, never on validation)."""
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=alpha,
                                                        learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                                        n_iter_no_change=6, random_state=seed))


# name -> (display name, factory(params, seed), parameter grid)
MODELS = {
    "persistence": ("Persistence", lambda p, s: Persistence(), {}),
    "ridge": ("Ridge regression", lambda p, s: _ridge(**p), {"alpha": [0.1, 1.0, 10.0, 100.0]}),
    "random_forest": ("Random forest", lambda p, s: _rf(seed=s, **p),
                      {"max_depth": [12, 20], "min_samples_leaf": [5, 20]}),
    "hist_gb": ("Gradient boosting (HistGB)", lambda p, s: _hgb(seed=s, **p),
                {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}),
    "mlp": ("Neural network (MLP)", lambda p, s: _mlp(seed=s, **p), {"alpha": [1e-4, 1e-3]}),
}


def grid(params: dict) -> list[dict]:
    if not params:
        return [{}]
    keys = list(params)
    return [dict(zip(keys, combo)) for combo in itertools.product(*params.values())]


def build(name: str, params: dict, seed: int):
    return MODELS[name][1](params, seed)
