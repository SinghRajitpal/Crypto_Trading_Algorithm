import numpy as np

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
