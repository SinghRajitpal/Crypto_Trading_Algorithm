import numpy as np
import pytest

import config
from backtest.ridge_layer import RidgeLayerSelector


class DummyDataEngine:
    def __init__(self, X, y, symbols):
        self._X = X
        self._y = y
        self._symbols = symbols

    def get_active_universe(self):
        return self._symbols

    def get_feature_matrix(self, symbol, window=None, exclude_outliers=True):
        if symbol not in self._symbols:
            return np.empty((0, 0)), np.empty((0,)), [], []
        return self._X, self._y, list(range(len(self._y))), [f"f{i}" for i in range(self._X.shape[1])]


def test_ridge_layer_selects_low_penalty_for_clean_linear():
    n = max(400, config.REGRESSION_MIN_TRAIN + 50)
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = 0.5 * x[:, 0] + 0.1
    de = DummyDataEngine(x, y, symbols=["A"])
    selector = RidgeLayerSelector(k_grid=[0.0, 10.0], train_min=20, val_len=10, t_threshold=0.0)
    res = selector.select(de)
    assert res.k_per_asset.get("A") == 0.0
    assert res.samples_per_asset.get("A") == n


def test_ridge_layer_intercept_not_penalized():
    n = 30
    X = np.empty((n, 0))
    X_int = RidgeLayerSelector._with_intercept(X)
    XtX = X_int.T @ X_int
    y = np.ones(n) * 0.5
    Xty = X_int.T @ y
    penalty = RidgeLayerSelector._penalty_matrix(XtX.shape[0], 1e6)
    beta = RidgeLayerSelector._ridge_beta(XtX, Xty, penalty)
    # Intercept should stay near mean(y) even with huge k
    assert beta.shape[0] == 1
    assert beta[0] == pytest.approx(0.5, rel=1e-3)


def test_ridge_layer_skips_insufficient_history():
    X = np.random.normal(size=(10, 2))
    y = np.random.normal(size=10)
    de = DummyDataEngine(X, y, symbols=["A"])
    selector = RidgeLayerSelector(k_grid=[0.1, 1.0], train_min=50, val_len=10, t_threshold=0.0)
    res = selector.select(de)
    assert "A" not in res.k_per_asset
