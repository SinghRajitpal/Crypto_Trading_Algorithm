import numpy as np

import config
from algorithm.forecast.ridge_regression import RidgeRegressionForecaster
from backtest.ridge_layer import RidgeLayerSelector


def test_forecast_respects_train_window_and_fixed_k(monkeypatch):
    # Ensure skip_cv path trains on exactly the requested window.
    monkeypatch.setattr(config, "REGRESSION_MIN_TRAIN", 5)
    monkeypatch.setattr(config, "REGRESSION_VAL_WINDOW", 5)
    forecaster = RidgeRegressionForecaster(k_grid=[0.5], t_threshold=0.0)
    n = 40
    X = np.stack([np.linspace(0.0, 1.0, n)], axis=1)
    y = np.linspace(0.0, 0.1, n)
    res = forecaster.forecast("AAA", X, y, fixed_k=0.5, train_window=10, skip_cv=True)
    assert res is not None
    assert res.samples == 10
    assert res.k_best == 0.5


def test_select_for_asset_handles_large_val_block(monkeypatch):
    # Disable mem logging to keep test output clean.
    monkeypatch.setattr(config, "LAYERA_DEBUG_MEM", False)
    monkeypatch.setattr(config, "REGRESSION_MIN_TRAIN", 10)
    n = 60
    X = np.stack([np.linspace(0.0, 1.0, n)], axis=1)
    y = np.linspace(0.0, 0.05, n)

    class DummyEngine:
        def get_feature_matrix(self, sym):
            return X, y, list(range(n)), ["f0"]

    selector = RidgeLayerSelector(
        k_grid=[0.1],
        train_min=20,
        val_len=500,  # deliberately oversized to exercise dynamic val sizing
        embargo=0,
        max_splits=5,
    )
    res = selector._select_for_asset("1h", "AAA", DummyEngine(), [25])
    assert res is not None
    sym_out, best_k, best_w, best_msep, rl, samples = res
    assert sym_out == "AAA"
    assert best_w == 25
    assert samples >= selector.train_min
