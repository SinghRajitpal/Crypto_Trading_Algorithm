import numpy as np
import pytest

import config
from algorithm.forecast.ridge_regression import RidgeRegressionForecaster


def test_ridge_forecaster_linear_relation():
    # y = 2*x + noise
    x = np.linspace(0, 10, 400)
    noise = np.random.normal(scale=0.1, size=400)
    y = 2 * x + noise
    X = x.reshape(-1, 1)
    forecaster = RidgeRegressionForecaster(k_grid=[1e-6, 1e-4, 1e-2], t_threshold=0.0)
    result = forecaster.forecast("SYM", X, y)
    assert result is not None
    # Expected log return should track roughly latest x * 2
    assert abs(result.expected_log_return - 2 * X[-1][0]) < 1.0
    assert result.msep >= 0
    assert result.gcv is not None
    assert result.hat_mean >= 0
    assert result.resid_sigma >= 0


def test_ridge_forecaster_requires_min_history():
    x = np.linspace(0, 1, 50)
    y = x * 0.1
    forecaster = RidgeRegressionForecaster(k_grid=[1e-2], t_threshold=0.0)
    result = forecaster.forecast("SYM", x.reshape(-1, 1), y)
    assert result is None  # below REGRESSION_MIN_TRAIN


def test_intercept_not_penalized_with_large_k():
    n = max(320, config.REGRESSION_MIN_TRAIN + 5)
    X = np.zeros((n, 1))
    true_intercept = 0.05
    y = np.full(n, true_intercept)
    forecaster = RidgeRegressionForecaster(k_grid=[1e4], t_threshold=0.0)
    result = forecaster.forecast("SYM", X, y)
    assert result is not None
    # If intercept were penalized heavily this would collapse toward 0
    assert result.expected_log_return == pytest.approx(true_intercept, rel=0.2)
    assert result.expected_simple_return == pytest.approx(np.exp(true_intercept) - 1, rel=0.2)


def test_t_stat_pruning_drops_weak_features():
    rng = np.random.default_rng(1)
    n = max(350, config.REGRESSION_MIN_TRAIN + 10)
    x_signal = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    y = 0.5 * x_signal + rng.normal(scale=0.1, size=n)
    X = np.column_stack([x_signal, x_noise])
    forecaster = RidgeRegressionForecaster(k_grid=[1e-3, 1e-2], t_threshold=1.5)
    result = forecaster.forecast("SYM", X, y)
    assert result is not None
    assert result.dropped_features  # noise feature should be flagged
    assert result.samples == n


def test_expected_simple_return_conversion():
    x = np.linspace(0, 5, 400)
    y = 0.1 * x  # positive slope to force positive log-return prediction
    forecaster = RidgeRegressionForecaster(k_grid=[1e-4], t_threshold=0.0)
    result = forecaster.forecast("SYM", x.reshape(-1, 1), y)
    assert result is not None
    assert result.expected_simple_return == pytest.approx(np.exp(result.expected_log_return) - 1)
